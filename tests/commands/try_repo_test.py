from __future__ import annotations

import os.path
import re
import time
from unittest import mock

import re_assert

from pre_commit import git
from pre_commit.commands.try_repo import try_repo
from pre_commit.store import Store
from pre_commit.util import cmd_output
from testing.auto_namedtuple import auto_namedtuple
from testing.fixtures import git_dir
from testing.fixtures import make_repo
from testing.fixtures import modify_manifest
from testing.util import cwd
from testing.util import git_commit
from testing.util import run_opts


def try_repo_opts(repo, ref=None, **kwargs):
    return auto_namedtuple(repo=repo, ref=ref, **run_opts(**kwargs)._asdict())


def _select_store_rows(store_dir):
    store = Store(store_dir)
    with store.connect() as db:
        return db.execute(
            'SELECT repo, ref, path FROM repos',
        ).fetchall()


def _get_out(cap_out):
    out = re.sub(r'\[INFO\].+\n', '', cap_out.get())
    start, using_config, config, rest = out.split(f'{"=" * 79}\n')
    assert using_config == 'Using config:\n'
    return start, config, rest


def _add_test_file():
    open('test-file', 'a').close()
    cmd_output('git', 'add', '.')


def _run_try_repo(mock_store_dir, tempdir_factory, **kwargs):
    repo = make_repo(tempdir_factory, 'modified_file_returns_zero_repo')
    with cwd(git_dir(tempdir_factory)):
        _add_test_file()
        assert not try_repo(try_repo_opts(repo, **kwargs))


def test_try_repo_repo_only(mock_store_dir, cap_out, tempdir_factory):
    with mock.patch.object(time, 'monotonic', return_value=0.0):
        _run_try_repo(mock_store_dir, tempdir_factory, verbose=True)
    start, config, rest = _get_out(cap_out)
    assert start == ''
    config_pattern = re_assert.Matches(
        '^repos:\n'
        '-   repo: .+\n'
        '    rev: .+\n'
        '    hooks:\n'
        '    -   id: bash_hook\n'
        '    -   id: bash_hook2\n'
        '    -   id: bash_hook3\n$',
    )
    config_pattern.assert_matches(config)
    assert rest == '''\
Bash hook............................................(no files to check)Skipped
- hook id: bash_hook
Bash hook................................................................Passed
- hook id: bash_hook2
- duration: 0s

test-file

Bash hook............................................(no files to check)Skipped
- hook id: bash_hook3
'''


def test_try_repo_with_specific_hook(
        mock_store_dir, cap_out, tempdir_factory,
):
    _run_try_repo(
        mock_store_dir, tempdir_factory, hook='bash_hook', verbose=True,
    )
    start, config, rest = _get_out(cap_out)
    assert start == ''
    config_pattern = re_assert.Matches(
        '^repos:\n'
        '-   repo: .+\n'
        '    rev: .+\n'
        '    hooks:\n'
        '    -   id: bash_hook\n$',
    )
    config_pattern.assert_matches(config)
    assert rest == '''\
Bash hook............................................(no files to check)Skipped
- hook id: bash_hook
'''


def test_try_repo_relative_path(mock_store_dir, cap_out, tempdir_factory):
    repo = make_repo(tempdir_factory, 'modified_file_returns_zero_repo')
    with cwd(git_dir(tempdir_factory)):
        _add_test_file()
        relative_repo = os.path.relpath(repo, '.')
        # previously crashed on cloning a relative path
        assert not try_repo(try_repo_opts(relative_repo, hook='bash_hook'))


def test_try_repo_bare_repo(mock_store_dir, cap_out, tempdir_factory):
    repo = make_repo(tempdir_factory, 'modified_file_returns_zero_repo')
    with cwd(git_dir(tempdir_factory)):
        _add_test_file()
        bare_repo = os.path.join(repo, '.git')
        # previously crashed attempting modification changes
        assert not try_repo(try_repo_opts(bare_repo, hook='bash_hook'))


def test_try_repo_specific_revision(mock_store_dir, cap_out, tempdir_factory):
    repo = make_repo(tempdir_factory, 'script_hooks_repo')
    ref = git.head_rev(repo)
    git_commit(cwd=repo)
    with cwd(git_dir(tempdir_factory)):
        _add_test_file()
        assert not try_repo(try_repo_opts(repo, ref=ref))

    _, config, _ = _get_out(cap_out)
    assert ref in config


def test_try_repo_uncommitted_changes(
        mock_store_dir, cap_out, tempdir_factory,
):
    repo = make_repo(tempdir_factory, 'script_hooks_repo')
    # make an uncommitted change
    with modify_manifest(repo, commit=False) as manifest:
        manifest[0]['name'] = 'modified name!'

    with cwd(git_dir(tempdir_factory)):
        open('test-fie', 'a').close()
        cmd_output('git', 'add', '.')
        assert not try_repo(try_repo_opts(repo))

    start, config, rest = _get_out(cap_out)
    assert start == '[WARNING] Creating temporary repo with uncommitted changes...\n'  # noqa: E501
    config_pattern = re_assert.Matches(
        '^repos:\n'
        '-   repo: .+try-repo/.+\n'
        '    rev: .+\n'
        '    hooks:\n'
        '    -   id: bash_hook\n$',
    )
    config_pattern.assert_matches(config)
    assert rest == 'modified name!...........................................................Passed\n'  # noqa: E501


def test_try_repo_staged_changes(mock_store_dir, tempdir_factory):
    repo = make_repo(tempdir_factory, 'modified_file_returns_zero_repo')

    with cwd(repo):
        open('staged-file', 'a').close()
        open('second-staged-file', 'a').close()
        cmd_output('git', 'add', '.')

    with cwd(git_dir(tempdir_factory)):
        assert not try_repo(try_repo_opts(repo, hook='bash_hook'))


def test_try_repo_reuses_environment_for_identical_changes(
        mock_store_dir, tempdir_factory, caplog,
):
    repo = make_repo(tempdir_factory, 'script_hooks_repo')
    # make an uncommitted change
    with modify_manifest(repo, commit=False) as manifest:
        manifest[0]['name'] = 'modified name!'

    with cwd(git_dir(tempdir_factory)):
        open('test-fie', 'a').close()
        cmd_output('git', 'add', '.')
        assert not try_repo(try_repo_opts(repo, hook='bash_hook'))
        first = _select_store_rows(mock_store_dir)

        # running again against the same content reuses the cache
        caplog.clear()
        assert not try_repo(try_repo_opts(repo, hook='bash_hook'))
        second = _select_store_rows(mock_store_dir)

    assert len(first) == 1
    assert 'try-repo' in first[0][0]
    assert first == second
    assert not any(
        msg.startswith('Initializing environment for ')
        for _, _, msg in caplog.record_tuples
    )


def test_try_repo_reinstalls_and_prunes_on_content_change(
        mock_store_dir, tempdir_factory,
):
    repo = make_repo(tempdir_factory, 'script_hooks_repo')

    with cwd(git_dir(tempdir_factory)):
        open('test-fie', 'a').close()
        cmd_output('git', 'add', '.')

        with modify_manifest(repo, commit=False) as manifest:
            manifest[0]['name'] = 'name one'
        assert not try_repo(try_repo_opts(repo, hook='bash_hook'))
        first = _select_store_rows(mock_store_dir)
        assert len(first) == 1
        old_name, old_ref, old_path = first[0]

        with modify_manifest(repo, commit=False) as manifest:
            manifest[0]['name'] = 'name two'
        assert not try_repo(try_repo_opts(repo, hook='bash_hook'))
        second = _select_store_rows(mock_store_dir)

    assert len(second) == 1
    new_name, new_ref, new_path = second[0]
    # same source repository keeps the same stable cache identity ...
    assert new_name == old_name
    # ... but changed content gets a new revision ...
    assert new_ref != old_ref
    # ... and the superseded environment is cleaned up
    assert os.path.exists(new_path)
    assert not os.path.exists(old_path)


def test_try_repo_does_not_modify_source_repo(
        mock_store_dir, tempdir_factory,
):
    repo = make_repo(tempdir_factory, 'script_hooks_repo')
    with modify_manifest(repo, commit=False) as manifest:
        manifest[0]['name'] = 'modified name!'

    head = git.head_rev(repo)
    _, status, _ = cmd_output('git', 'status', '--porcelain', cwd=repo)

    with cwd(git_dir(tempdir_factory)):
        open('test-fie', 'a').close()
        cmd_output('git', 'add', '.')
        assert not try_repo(try_repo_opts(repo, hook='bash_hook'))

    assert git.head_rev(repo) == head
    _, status_after, _ = cmd_output(
        'git', 'status', '--porcelain', cwd=repo,
    )
    assert status_after == status


def test_try_repo_explicit_ref_does_not_use_shadow(
        mock_store_dir, tempdir_factory,
):
    repo = make_repo(tempdir_factory, 'script_hooks_repo')
    ref = git.head_rev(repo)
    # the working tree has uncommitted changes which must be ignored
    with modify_manifest(repo, commit=False) as manifest:
        manifest[0]['name'] = 'modified name!'

    with cwd(git_dir(tempdir_factory)):
        open('test-fie', 'a').close()
        cmd_output('git', 'add', '.')
        assert not try_repo(
            try_repo_opts(repo, ref=ref, hook='bash_hook'),
        )

    assert not os.path.exists(os.path.join(mock_store_dir, 'try-repo'))
    rows = _select_store_rows(mock_store_dir)
    assert rows == [(repo, ref, rows[0][2])]
    assert os.path.exists(rows[0][2])
