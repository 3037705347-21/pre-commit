from __future__ import annotations

import hashlib
import logging
import os.path
import shlex
import sqlite3
import stat
import subprocess
import sys
from unittest import mock

import pytest

import pre_commit.constants as C
from pre_commit import git
from pre_commit import main
from pre_commit.commands.gc import gc
from pre_commit.errors import FatalError
from pre_commit.store import _get_default_directory
from pre_commit.store import _LOCAL_RESOURCES
from pre_commit.store import _SCHEMA_MIGRATIONS
from pre_commit.store import CURRENT_SCHEMA_VERSION
from pre_commit.store import Store
from pre_commit.util import CalledProcessError
from pre_commit.util import cmd_output
from testing.fixtures import git_dir
from testing.util import cwd
from testing.util import git_commit
from testing.util import xfailif_windows


def _select_all_configs(store: Store) -> list[str]:
    with store.connect() as db:
        rows = db.execute('SELECT * FROM configs').fetchall()
        return [path for path, in rows]


def _select_all_repos(store: Store) -> list[tuple[str, str, str]]:
    with store.connect() as db:
        return db.execute('SELECT repo, ref, path FROM repos').fetchall()


def test_our_session_fixture_works():
    """There's a session fixture which makes `Store` invariantly raise to
    prevent writing to the home directory.
    """
    with pytest.raises(AssertionError):
        Store()


def test_get_default_directory_defaults_to_home():
    # Not we use the module level one which is not mocked
    ret = _get_default_directory()
    expected = os.path.realpath(os.path.expanduser('~/.cache/pre-commit'))
    assert ret == expected


def test_adheres_to_xdg_specification():
    with mock.patch.dict(
        os.environ, {'XDG_CACHE_HOME': '/tmp/fakehome'},
    ):
        ret = _get_default_directory()
    expected = os.path.realpath('/tmp/fakehome/pre-commit')
    assert ret == expected


def test_uses_environment_variable_when_present():
    with mock.patch.dict(
        os.environ, {'PRE_COMMIT_HOME': '/tmp/pre_commit_home'},
    ):
        ret = _get_default_directory()
    expected = os.path.realpath('/tmp/pre_commit_home')
    assert ret == expected


def test_store_init(store):
    # Should create the store directory
    assert os.path.exists(store.directory)
    # Should create a README file indicating what the directory is about
    with open(os.path.join(store.directory, 'README')) as readme_file:
        readme_contents = readme_file.read()
        for text_line in (
            'This directory is maintained by the pre-commit project.',
            'Learn more: https://github.com/pre-commit/pre-commit',
        ):
            assert text_line in readme_contents


def test_clone(store, tempdir_factory, caplog):
    path = git_dir(tempdir_factory)
    with cwd(path):
        git_commit()
        rev = git.head_rev(path)
        git_commit()

    ret = store.clone(path, rev)
    # Should have printed some stuff
    assert caplog.record_tuples[0][-1].startswith(
        'Initializing environment for ',
    )

    # Should return a directory inside of the store
    assert os.path.exists(ret)
    assert ret.startswith(store.directory)
    # Directory should start with `repo`
    _, dirname = os.path.split(ret)
    assert dirname.startswith('repo')
    # Should be checked out to the rev we specified
    assert git.head_rev(ret) == rev

    # Assert there's an entry in the sqlite db for this
    assert _select_all_repos(store) == [(path, rev, ret)]


def test_warning_for_deprecated_stages_on_init(store, tempdir_factory, caplog):
    manifest = '''\
-   id: hook1
    name: hook1
    language: system
    entry: echo hook1
    stages: [commit, push]
-   id: hook2
    name: hook2
    language: system
    entry: echo hook2
    stages: [push, merge-commit]
'''

    path = git_dir(tempdir_factory)
    with open(os.path.join(path, C.MANIFEST_FILE), 'w') as f:
        f.write(manifest)
    cmd_output('git', 'add', '.', cwd=path)
    git_commit(cwd=path)
    rev = git.head_rev(path)

    store.clone(path, rev)
    assert caplog.record_tuples[1] == (
        'pre_commit',
        logging.WARNING,
        f'repo `{path}` uses deprecated stage names '
        f'(commit, push, merge-commit) which will be removed in a future '
        f'version.  '
        f'Hint: often `pre-commit autoupdate --repo {shlex.quote(path)}` '
        f'will fix this.  '
        f'if it does not -- consider reporting an issue to that repo.',
    )

    # should not re-warn
    caplog.clear()
    store.clone(path, rev)
    assert caplog.record_tuples == []


def test_no_warning_for_non_deprecated_stages_on_init(
        store, tempdir_factory, caplog,
):
    manifest = '''\
-   id: hook1
    name: hook1
    language: system
    entry: echo hook1
    stages: [pre-commit, pre-push]
-   id: hook2
    name: hook2
    language: system
    entry: echo hook2
    stages: [pre-push, pre-merge-commit]
'''

    path = git_dir(tempdir_factory)
    with open(os.path.join(path, C.MANIFEST_FILE), 'w') as f:
        f.write(manifest)
    cmd_output('git', 'add', '.', cwd=path)
    git_commit(cwd=path)
    rev = git.head_rev(path)

    store.clone(path, rev)
    assert logging.WARNING not in {tup[1] for tup in caplog.record_tuples}


def test_clone_cleans_up_on_checkout_failure(store):
    with pytest.raises(Exception) as excinfo:
        # This raises an exception because you can't clone something that
        # doesn't exist!
        store.clone('/i_dont_exist_lol', 'fake_rev')
    assert '/i_dont_exist_lol' in str(excinfo.value)

    repo_dirs = [
        d for d in os.listdir(store.directory) if d.startswith('repo')
    ]
    assert repo_dirs == []


def test_clone_when_repo_already_exists(store):
    # Create an entry in the sqlite db that makes it look like the repo has
    # been cloned.
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            'INSERT INTO repos (repo, ref, path) '
            'VALUES ("fake_repo", "fake_ref", "fake_path")',
        )

    assert store.clone('fake_repo', 'fake_ref') == 'fake_path'


def test_clone_shallow_failure_fallback_to_complete(
    store, tempdir_factory,
    caplog,
):
    path = git_dir(tempdir_factory)
    with cwd(path):
        git_commit()
        rev = git.head_rev(path)
        git_commit()

    # Force shallow clone failure
    def fake_shallow_clone(self, *args, **kwargs):
        raise CalledProcessError(1, (), b'', None)
    store._shallow_clone = fake_shallow_clone

    ret = store.clone(path, rev)

    # Should have printed some stuff
    assert caplog.record_tuples[0][-1].startswith(
        'Initializing environment for ',
    )

    # Should return a directory inside of the store
    assert os.path.exists(ret)
    assert ret.startswith(store.directory)
    # Directory should start with `repo`
    _, dirname = os.path.split(ret)
    assert dirname.startswith('repo')
    # Should be checked out to the rev we specified
    assert git.head_rev(ret) == rev

    # Assert there's an entry in the sqlite db for this
    assert _select_all_repos(store) == [(path, rev, ret)]


def test_clone_tag_not_on_mainline(store, tempdir_factory):
    path = git_dir(tempdir_factory)
    with cwd(path):
        git_commit()
        cmd_output('git', 'checkout', 'master', '-b', 'branch')
        git_commit()
        cmd_output('git', 'tag', 'v1')
        cmd_output('git', 'checkout', 'master')
        cmd_output('git', 'branch', '-D', 'branch')

    # previously crashed on unreachable refs
    store.clone(path, 'v1')


def test_create_when_directory_exists_but_not_db(store):
    # In versions <= 0.3.5, there was no sqlite db causing a need for
    # backward compatibility
    os.remove(store.db_path)
    store = Store(store.directory)
    assert os.path.exists(store.db_path)


def test_create_when_store_already_exists(store):
    # an assertion that this is idempotent and does not crash
    Store(store.directory)


def test_db_repo_name(store):
    assert store.db_repo_name('repo', ()) == 'repo'
    assert store.db_repo_name('repo', ('b', 'a', 'c')) == 'repo:b,a,c'


def test_local_resources_reflects_reality():
    on_disk = {
        res.removeprefix('empty_template_')
        for res in os.listdir('pre_commit/resources')
        if res.startswith('empty_template_')
    }
    assert on_disk == {os.path.basename(x) for x in _LOCAL_RESOURCES}


def test_mark_config_as_used(store, tmpdir):
    with tmpdir.as_cwd():
        f = tmpdir.join('f').ensure()
        store.mark_config_used('f')
        assert _select_all_configs(store) == [f.strpath]


def test_mark_config_as_used_idempotent(store, tmpdir):
    test_mark_config_as_used(store, tmpdir)
    test_mark_config_as_used(store, tmpdir)


def test_mark_config_as_used_does_not_exist(store):
    store.mark_config_used('f')
    assert _select_all_configs(store) == []


def test_mark_config_as_used_roll_forward(store, tmpdir):
    with store.connect() as db:  # simulate pre-1.14.0
        db.executescript('DROP TABLE configs')
    test_mark_config_as_used(store, tmpdir)


@xfailif_windows  # pragma: win32 no cover
def test_mark_config_as_used_readonly(tmpdir):
    cfg = tmpdir.join('f').ensure()
    store_dir = tmpdir.join('store')
    # make a store, then we'll convert its directory to be readonly
    assert not Store(str(store_dir)).readonly  # directory didn't exist
    assert not Store(str(store_dir)).readonly  # directory did exist

    def _chmod_minus_w(p):
        st = os.stat(p)
        os.chmod(p, st.st_mode & ~(stat.S_IWUSR | stat.S_IWOTH | stat.S_IWGRP))

    _chmod_minus_w(store_dir)
    for fname in os.listdir(store_dir):
        assert not os.path.isdir(fname)
        _chmod_minus_w(os.path.join(store_dir, fname))

    store = Store(str(store_dir))
    assert store.readonly
    # should be skipped due to readonly
    store.mark_config_used(str(cfg))
    assert _select_all_configs(store) == []


def test_clone_with_recursive_submodules(store, tmp_path):
    sub = tmp_path.joinpath('sub')
    sub.mkdir()
    sub.joinpath('submodule').write_text('i am a submodule')
    cmd_output('git', '-C', str(sub), 'init', '.')
    cmd_output('git', '-C', str(sub), 'add', '.')
    git.commit(str(sub))

    repo = tmp_path.joinpath('repo')
    repo.mkdir()
    repo.joinpath('repository').write_text('i am a repo')
    cmd_output('git', '-C', str(repo), 'init', '.')
    cmd_output('git', '-C', str(repo), 'add', '.')
    cmd_output('git', '-C', str(repo), 'submodule', 'add', str(sub), 'sub')
    git.commit(str(repo))

    rev = git.head_rev(str(repo))
    ret = store.clone(str(repo), rev)

    assert os.path.exists(ret)
    assert os.path.exists(os.path.join(ret, str(repo), 'repository'))
    assert os.path.exists(os.path.join(ret, str(sub), 'submodule'))


# ---------------------------------------------------------------------------
# schema version reconciliation

_LEGACY_REPO_ROWS = [
    ('https://example.com/repo', 'v1', '/cache/repo1'),
    ('local', C.LOCAL_REPO_VERSION, '/cache/repo2'),
]
_LEGACY_CONFIG_ROWS = [('/home/user/.pre-commit-config.yaml')]


def _user_version(db_path: str) -> int:
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
        return db.execute('PRAGMA user_version').fetchone()[0]


def _table_names(db_path: str) -> set[str]:
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
        return {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'",
            )
        }


def _write_legacy_db(home: str, *, with_configs: bool) -> str:
    """Write a database as produced before schema versioning existed
    (`user_version == 0`)."""
    os.makedirs(home)
    db_path = os.path.join(home, 'db.db')
    with sqlite3.connect(db_path) as db:
        db.execute(
            'CREATE TABLE repos ('
            '    repo TEXT NOT NULL,'
            '    ref TEXT NOT NULL,'
            '    path TEXT NOT NULL,'
            '    PRIMARY KEY (repo, ref)'
            ')',
        )
        db.executemany('INSERT INTO repos VALUES (?, ?, ?)', _LEGACY_REPO_ROWS)
        if with_configs:
            db.execute(
                'CREATE TABLE configs ('
                '   path TEXT NOT NULL,'
                '   PRIMARY KEY (path)'
                ')',
            )
            db.executemany(
                'INSERT INTO configs VALUES (?)',
                [(p,) for p in _LEGACY_CONFIG_ROWS],
            )
    assert _user_version(db_path) == 0
    return db_path


def _digest(path: str) -> tuple[int, int, bytes]:
    with open(path, 'rb') as f:
        return (
            os.stat(path).st_size,
            os.stat(path).st_mtime_ns,
            hashlib.sha256(f.read()).digest(),
        )


def test_fresh_db_records_current_schema_version(store):
    assert _user_version(store.db_path) == CURRENT_SCHEMA_VERSION
    # the historical table structure is created unchanged
    with store.connect() as db:
        repos_cols = [r[1] for r in db.execute('PRAGMA table_info(repos)')]
        configs_cols = [
            r[1] for r in db.execute('PRAGMA table_info(configs)')
        ]
    assert repos_cols == ['repo', 'ref', 'path']
    assert configs_cols == ['path']


def test_legacy_db_without_configs_is_migrated_and_rows_kept(tmpdir):
    home = tmpdir.join('store').strpath
    db_path = _write_legacy_db(home, with_configs=False)

    store = Store(home)

    assert _user_version(db_path) == CURRENT_SCHEMA_VERSION
    assert 'configs' in _table_names(db_path)
    assert _select_all_repos(store) == _LEGACY_REPO_ROWS
    assert _select_all_configs(store) == []


def test_legacy_db_with_configs_is_stamped_and_rows_kept(tmpdir):
    home = tmpdir.join('store').strpath
    db_path = _write_legacy_db(home, with_configs=True)

    store = Store(home)

    assert _user_version(db_path) == CURRENT_SCHEMA_VERSION
    assert _select_all_repos(store) == _LEGACY_REPO_ROWS
    assert _select_all_configs(store) == _LEGACY_CONFIG_ROWS


def test_opening_current_version_db_does_not_write(store):
    before = _digest(store.db_path)
    Store(store.directory)
    Store(store.directory)
    assert _digest(store.db_path) == before


def test_future_schema_version_is_fatal_and_untouched(store):
    with sqlite3.connect(store.db_path) as db:
        db.execute(f'PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}')
        db.commit()
    before = _digest(store.db_path)

    with pytest.raises(FatalError) as excinfo:
        Store(store.directory)

    msg = str(excinfo.value)
    assert str(CURRENT_SCHEMA_VERSION + 1) in msg
    assert store.db_path in msg
    assert _digest(store.db_path) == before
    assert _user_version(store.db_path) == CURRENT_SCHEMA_VERSION + 1


def test_future_schema_version_cli_exits_nonzero(mock_store_dir, cap_out):
    Store()
    db_path = os.path.join(mock_store_dir, 'db.db')
    with sqlite3.connect(db_path) as db:
        db.execute('PRAGMA user_version = 99')
        db.commit()

    with pytest.raises(SystemExit) as excinfo:
        main.main(('gc',))
    assert excinfo.value.code == 1

    out = cap_out.get()
    assert 'An error has occurred: FatalError' in out
    assert 'schema version 99' in out
    assert _user_version(db_path) == 99


def test_clean_still_works_on_future_schema(mock_store_dir, cap_out, tmpdir):
    Store()
    db_path = os.path.join(mock_store_dir, 'db.db')
    with sqlite3.connect(db_path) as db:
        db.execute('PRAGMA user_version = 99')
        db.commit()

    fake_legacy = tmpdir.join('legacy').strpath
    os.makedirs(fake_legacy)
    with mock.patch.object(
            os.path, 'expanduser', return_value=fake_legacy,
    ):
        assert main.main(('clean',)) == 0
    assert not os.path.exists(db_path)
    assert cap_out.get().startswith('Cleaned ')


def test_unrecognized_database_is_not_modified(tmpdir):
    home = tmpdir.join('store').strpath
    os.makedirs(home)
    db_path = os.path.join(home, 'db.db')
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE unrelated (thing TEXT)')
        db.commit()
    before = _digest(db_path)

    with pytest.raises(FatalError):
        Store(home)
    assert _digest(db_path) == before


def test_failed_migration_rolls_back_and_rerun_converges(tmpdir):
    home = tmpdir.join('store').strpath
    db_path = _write_legacy_db(home, with_configs=False)

    def _fail_after_ddl(db):
        db.execute(
            'CREATE TABLE IF NOT EXISTS configs ('
            '   path TEXT NOT NULL,'
            '   PRIMARY KEY (path)'
            ')',
        )
        raise RuntimeError('simulated failure mid-migration')

    with mock.patch.dict(_SCHEMA_MIGRATIONS, {1: _fail_after_ddl}):
        with pytest.raises(RuntimeError):
            Store(home)

    # the half-applied migration was rolled back -- nothing is committed
    assert _user_version(db_path) == 0
    assert 'configs' not in _table_names(db_path)
    with sqlite3.connect(db_path) as db:
        repos_rows = db.execute('SELECT * FROM repos').fetchall()
        assert repos_rows == _LEGACY_REPO_ROWS

    # running again (with the real migration) converges
    store = Store(home)
    assert _user_version(db_path) == CURRENT_SCHEMA_VERSION
    assert _select_all_repos(store) == _LEGACY_REPO_ROWS
    assert 'configs' in _table_names(db_path)


@xfailif_windows  # pragma: win32 no cover
def test_killed_migration_is_finished_after_restart(tmpdir):
    home = tmpdir.join('store').strpath
    db_path = _write_legacy_db(home, with_configs=False)
    repo_root = os.path.dirname(os.path.dirname(__file__))
    env = {
        **os.environ,
        'PC_HOME': home,
        'PC_REPO': repo_root,
        'PYTHONPATH': (
            repo_root + os.pathsep + os.environ.get('PYTHONPATH', '')
        ),
    }
    crash_code = (
        'import os\n'
        'from pre_commit.store import Store, _SCHEMA_MIGRATIONS\n'
        'def _crash(db):\n'
        '    db.execute(\n'
        "        'CREATE TABLE IF NOT EXISTS configs ('\n"
        "        ' path TEXT NOT NULL, PRIMARY KEY (path))'\n"
        '    )\n'
        '    os._exit(7)\n'
        '_SCHEMA_MIGRATIONS[1] = _crash\n'
        "Store(os.environ['PC_HOME'])\n"
    )
    killed = subprocess.run(
        (sys.executable, '-c', crash_code),
        env=env, capture_output=True, text=True,
    )
    assert killed.returncode == 7
    # the uncommitted migration left a hot journal behind
    assert os.path.exists(db_path + '-journal')
    assert _user_version(db_path) == 0

    # restart in a fresh process: recovery + migration completes
    resume_code = (
        'import os\n'
        'from pre_commit.store import Store\n'
        'Store(os.environ["PC_HOME"])\n'
    )
    resumed = subprocess.run(
        (sys.executable, '-c', resume_code),
        env=env, capture_output=True, text=True,
    )
    assert resumed.returncode == 0, resumed.stderr

    assert not os.path.exists(db_path + '-journal')
    assert _user_version(db_path) == CURRENT_SCHEMA_VERSION
    assert 'configs' in _table_names(db_path)
    with sqlite3.connect(db_path) as db:
        repos_rows = db.execute('SELECT * FROM repos').fetchall()
        assert repos_rows == _LEGACY_REPO_ROWS


def test_gc_and_mark_config_used_work_after_migration(tmpdir, cap_out):
    home = tmpdir.join('store').strpath
    db_path = _write_legacy_db(home, with_configs=False)
    # remove rows so gc has nothing on disk to delete
    with sqlite3.connect(db_path) as db:
        db.execute('DELETE FROM repos')
        db.commit()

    store = Store(home)
    assert gc(store) == 0
    assert cap_out.get() == '0 repo(s) removed.\n'

    cfg = tmpdir.join('cfg.yaml').ensure()
    store.mark_config_used(cfg.strpath)
    assert _select_all_configs(store) == [cfg.strpath]


@xfailif_windows  # pragma: win32 no cover
def test_readonly_legacy_cache_is_not_modified(tmpdir):
    home = tmpdir.join('store').strpath
    db_path = _write_legacy_db(home, with_configs=False)
    write = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    os.chmod(db_path, os.stat(db_path).st_mode & ~write)
    os.chmod(home, os.stat(home).st_mode & ~write)
    try:
        with pytest.raises(FatalError):
            Store(home)
        assert _user_version(db_path) == 0
        assert 'configs' not in _table_names(db_path)
    finally:
        os.chmod(home, os.stat(home).st_mode | stat.S_IWUSR)
        os.chmod(db_path, os.stat(db_path).st_mode | stat.S_IWUSR)


def test_passive_open_never_writes_or_migrates(tmpdir):
    """counter-proof: merely seeing that the schema is behind (the read-only
    version probe, or opening with reconciliation disabled) must not be
    mistaken for completing the migration."""
    home = tmpdir.join('store').strpath
    db_path = _write_legacy_db(home, with_configs=False)
    before = _digest(db_path)

    store = Store(home, reconcile_schema=False)
    assert store._read_schema_versions() == (0, 1)

    assert _user_version(db_path) == 0
    assert 'configs' not in _table_names(db_path)
    assert _digest(db_path) == before
