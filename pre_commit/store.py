from __future__ import annotations

import contextlib
import logging
import os.path
import sqlite3
import tempfile
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Sequence

import pre_commit.constants as C
from pre_commit import clientlib
from pre_commit import file_lock
from pre_commit import git
from pre_commit.errors import FatalError
from pre_commit.util import CalledProcessError
from pre_commit.util import clean_path_on_failure
from pre_commit.util import cmd_output_b
from pre_commit.util import resource_text


logger = logging.getLogger('pre_commit')


#: schema version of the databases created and understood by this version
#: of pre-commit.  the version is recorded in the database header via the
#: sqlite `user_version` pragma.
CURRENT_SCHEMA_VERSION = 2


def _migrate_1_to_2(db: sqlite3.Connection) -> None:
    # the `configs` table was added in schema version 2.  it was
    # historically created lazily, so old caches may not have it.
    db.execute(
        'CREATE TABLE IF NOT EXISTS configs ('
        '   path TEXT NOT NULL,'
        '   PRIMARY KEY (path)'
        ')',
    )


#: migrations indexed by the schema version they upgrade *from*.
_SCHEMA_MIGRATIONS = {
    1: _migrate_1_to_2,
}


def _get_default_directory() -> str:
    """Returns the default directory for the Store.  This is intentionally
    underscored to indicate that `Store.get_default_directory` is the intended
    way to get this information.  This is also done so
    `Store.get_default_directory` can be mocked in tests and
    `_get_default_directory` can be tested.
    """
    ret = os.environ.get('PRE_COMMIT_HOME') or os.path.join(
        os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
        'pre-commit',
    )
    return os.path.realpath(ret)


_LOCAL_RESOURCES = (
    'Cargo.toml', 'main.go', 'go.mod', 'main.rs',
    'pre-commit-package-dev-1.rockspec',
    'pre_commit_placeholder_package.gemspec', 'setup.py',
    'environment.yml', 'Makefile.PL', 'pubspec.yaml',
    'renv.lock', 'renv/activate.R', 'renv/LICENSE.renv',
)


def _make_local_repo(directory: str) -> None:
    for resource in _LOCAL_RESOURCES:
        resource_dirname, resource_basename = os.path.split(resource)
        contents = resource_text(f'empty_template_{resource_basename}')
        target_dir = os.path.join(directory, resource_dirname)
        target_file = os.path.join(target_dir, resource_basename)
        os.makedirs(target_dir, exist_ok=True)
        with open(target_file, 'w') as f:
            f.write(contents)


class Store:
    get_default_directory = staticmethod(_get_default_directory)

    def __init__(
            self,
            directory: str | None = None,
            *,
            reconcile_schema: bool = True,
    ) -> None:
        self.directory = directory or Store.get_default_directory()
        self.db_path = os.path.join(self.directory, 'db.db')
        self.readonly = (
            os.path.exists(self.directory) and
            not os.access(self.directory, os.W_OK)
        )

        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)
            with open(os.path.join(self.directory, 'README'), 'w') as f:
                f.write(
                    'This directory is maintained by the pre-commit project.\n'
                    'Learn more: https://github.com/pre-commit/pre-commit\n',
                )

        if os.path.exists(self.db_path):
            if reconcile_schema:
                self._reconcile_schema()
            return
        with self.exclusive_lock():
            # Another process may have already completed this work
            if os.path.exists(self.db_path):  # pragma: no cover (race)
                if reconcile_schema:
                    self._reconcile_schema()
                return
            # To avoid a race where someone ^Cs between db creation and
            # execution of the CREATE TABLE statement
            fd, tmpfile = tempfile.mkstemp(dir=self.directory)
            # We'll be managing this file ourselves
            os.close(fd)
            with self.connect(db_path=tmpfile) as db:
                db.executescript(
                    f'PRAGMA user_version = {CURRENT_SCHEMA_VERSION};\n'
                    'CREATE TABLE repos ('
                    '    repo TEXT NOT NULL,'
                    '    ref TEXT NOT NULL,'
                    '    path TEXT NOT NULL,'
                    '    PRIMARY KEY (repo, ref)'
                    ');',
                )
                self._create_configs_table(db)

            # Atomic file move
            os.replace(tmpfile, self.db_path)

    def _future_schema_error(self, version: int) -> FatalError:
        return FatalError(
            f'the pre-commit cache at {self.db_path} was written by a '
            f'newer version of pre-commit (database schema version '
            f'{version}, this version understands at most '
            f'{CURRENT_SCHEMA_VERSION}).  refusing to modify it -- upgrade '
            f'pre-commit or remove the cache directory ({self.directory})',
        )

    def _read_schema_versions(self) -> tuple[int, int]:
        """Return ``(recorded, effective)`` versions without writing.

        ``recorded`` is the version in the sqlite `user_version` header.
        Databases written before schema versioning existed record no
        version (``recorded == 0``); their effective version is inferred
        from the tables they contain.
        """
        with contextlib.closing(
            sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True),
        ) as db:
            recorded = db.execute('PRAGMA user_version').fetchone()[0]
            if recorded != 0:
                return recorded, recorded
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'",
                )
            }
            if 'repos' not in tables:
                raise FatalError(
                    f'unrecognized database at {self.db_path}: it is not '
                    f'a pre-commit cache (missing the `repos` table)',
                )
            # the `configs` table was introduced in schema version 2
            effective = 2 if 'configs' in tables else 1
            return 0, effective

    def _reconcile_schema(self) -> None:
        """Bring the cache database up to date with this pre-commit version.

        * a current database is left completely untouched (no writes)
        * a database written by a newer pre-commit is never modified; a
          FatalError is raised instead (the command exits non-zero)
        * an older database is migrated forward, preserving every row;
          migration runs inside a single sqlite transaction, so being
          interrupted (or failing) leaves the database in its
          pre-migration state and re-running simply continues
        """
        recorded, version = self._read_schema_versions()
        if version > CURRENT_SCHEMA_VERSION:
            raise self._future_schema_error(version)
        if recorded == CURRENT_SCHEMA_VERSION:
            return
        if self.readonly:  # pragma: win32 no cover
            raise FatalError(
                f'the pre-commit cache at {self.db_path} uses database '
                f'schema version {version}, but the cache directory is not '
                f'writable so it cannot be migrated to version '
                f'{CURRENT_SCHEMA_VERSION}',
            )

        with self.exclusive_lock():
            # another process may have finished the migration while we
            # waited for the lock
            recorded, version = self._read_schema_versions()
            if version > CURRENT_SCHEMA_VERSION:  # pragma: no cover (race)
                raise self._future_schema_error(version)
            if recorded == CURRENT_SCHEMA_VERSION:  # pragma: no cover (race)
                return

            with contextlib.closing(sqlite3.connect(self.db_path)) as db:
                # the transaction is managed explicitly: python's sqlite3
                # legacy transaction control does not auto-open
                # transactions around DDL, so a plain `with db:` block
                # would commit the migration statement by statement.
                db.isolation_level = None
                db.execute('BEGIN IMMEDIATE')
                try:
                    while version < CURRENT_SCHEMA_VERSION:
                        _SCHEMA_MIGRATIONS[version](db)
                        version += 1
                    # record the converged version.  this also stamps
                    # databases which already had the current tables but
                    # were written before versioning was recorded
                    db.execute(f'PRAGMA user_version = {version}')
                except BaseException:
                    db.execute('ROLLBACK')
                    raise
                else:
                    db.execute('COMMIT')

    @contextlib.contextmanager
    def exclusive_lock(self) -> Generator[None]:
        def blocked_cb() -> None:  # pragma: no cover (tests are in-process)
            logger.info('Locking pre-commit directory')

        with file_lock.lock(os.path.join(self.directory, '.lock'), blocked_cb):
            yield

    @contextlib.contextmanager
    def connect(
            self,
            db_path: str | None = None,
    ) -> Generator[sqlite3.Connection]:
        db_path = db_path or self.db_path
        # sqlite doesn't close its fd with its contextmanager >.<
        # contextlib.closing fixes this.
        # See: https://stackoverflow.com/a/28032829/812183
        with contextlib.closing(sqlite3.connect(db_path)) as db:
            # this creates a transaction
            with db:
                yield db

    @classmethod
    def db_repo_name(cls, repo: str, deps: Sequence[str]) -> str:
        if deps:
            return f'{repo}:{",".join(deps)}'
        else:
            return repo

    def _new_repo(
            self,
            repo: str,
            ref: str,
            deps: Sequence[str],
            make_strategy: Callable[[str], None],
    ) -> str:
        original_repo = repo
        repo = self.db_repo_name(repo, deps)

        def _get_result() -> str | None:
            # Check if we already exist
            with self.connect() as db:
                result = db.execute(
                    'SELECT path FROM repos WHERE repo = ? AND ref = ?',
                    (repo, ref),
                ).fetchone()
                return result[0] if result else None

        result = _get_result()
        if result:
            return result
        with self.exclusive_lock():
            # Another process may have already completed this work
            result = _get_result()
            if result:  # pragma: no cover (race)
                return result

            logger.info(f'Initializing environment for {repo}.')

            directory = tempfile.mkdtemp(prefix='repo', dir=self.directory)
            with clean_path_on_failure(directory):
                make_strategy(directory)

            # Update our db with the created repo
            with self.connect() as db:
                db.execute(
                    'INSERT INTO repos (repo, ref, path) VALUES (?, ?, ?)',
                    [repo, ref, directory],
                )

            clientlib.warn_for_stages_on_repo_init(original_repo, directory)

        return directory

    def _complete_clone(self, ref: str, git_cmd: Callable[..., None]) -> None:
        """Perform a complete clone of a repository and its submodules """

        git_cmd('fetch', 'origin', '--tags')
        git_cmd('checkout', ref)
        git_cmd('submodule', 'update', '--init', '--recursive')

    def _shallow_clone(self, ref: str, git_cmd: Callable[..., None]) -> None:
        """Perform a shallow clone of a repository and its submodules """

        git_config = 'protocol.version=2'
        git_cmd('-c', git_config, 'fetch', 'origin', ref, '--depth=1')
        git_cmd('checkout', 'FETCH_HEAD')
        git_cmd(
            '-c', git_config, 'submodule', 'update', '--init', '--recursive',
            '--depth=1',
        )

    def clone(self, repo: str, ref: str, deps: Sequence[str] = ()) -> str:
        """Clone the given url and checkout the specific ref."""

        def clone_strategy(directory: str) -> None:
            git.init_repo(directory, repo)
            env = git.no_git_env()

            def _git_cmd(*args: str) -> None:
                cmd_output_b('git', *args, cwd=directory, env=env)

            try:
                self._shallow_clone(ref, _git_cmd)
            except CalledProcessError:
                self._complete_clone(ref, _git_cmd)

        return self._new_repo(repo, ref, deps, clone_strategy)

    def make_local(self, deps: Sequence[str]) -> str:
        return self._new_repo(
            'local', C.LOCAL_REPO_VERSION, deps, _make_local_repo,
        )

    def _create_configs_table(self, db: sqlite3.Connection) -> None:
        db.executescript(
            'CREATE TABLE IF NOT EXISTS configs ('
            '   path TEXT NOT NULL,'
            '   PRIMARY KEY (path)'
            ');',
        )

    def mark_config_used(self, path: str) -> None:
        if self.readonly:  # pragma: win32 no cover
            return
        path = os.path.realpath(path)
        # don't insert config files that do not exist
        if not os.path.exists(path):
            return
        with self.connect() as db:
            # TODO: eventually remove this and only create in _create
            self._create_configs_table(db)
            db.execute('INSERT OR IGNORE INTO configs VALUES (?)', (path,))
