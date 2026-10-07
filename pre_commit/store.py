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


# the schema version understood by this release, stored in the database via
# `PRAGMA user_version`.
# a database at version 0 was written by a release which never recorded a
# version.  when the on-disk structure changes, bump this and register the
# upgrade in `_MIGRATIONS` (keyed by the version it upgrades to).
SCHEMA_VERSION = 1


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

    def __init__(self, directory: str | None = None) -> None:
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
            self._reconcile_schema()
            return
        with self.exclusive_lock():
            # Another process may have already completed this work
            if os.path.exists(self.db_path):  # pragma: no cover (race)
                self._reconcile_schema()
                return
            # To avoid a race where someone ^Cs between db creation and
            # execution of the CREATE TABLE statement
            fd, tmpfile = tempfile.mkstemp(dir=self.directory)
            # We'll be managing this file ourselves
            os.close(fd)
            with self.connect(db_path=tmpfile) as db:
                db.executescript(
                    'CREATE TABLE repos ('
                    '    repo TEXT NOT NULL,'
                    '    ref TEXT NOT NULL,'
                    '    path TEXT NOT NULL,'
                    '    PRIMARY KEY (repo, ref)'
                    ');',
                )
                self._create_configs_table(db)
                db.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')

            # Atomic file move
            os.replace(tmpfile, self.db_path)

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

    @staticmethod
    def _get_user_version(db: sqlite3.Connection) -> int:
        return db.execute('PRAGMA user_version').fetchone()[0]

    def _newer_version_error(self, version: int) -> FatalError:
        return FatalError(
            f'the cache database {self.db_path} was written by a newer '
            f'version of pre-commit (schema version {version}, but this '
            f'release understands up to {SCHEMA_VERSION}).  refusing to '
            f'modify it -- upgrade pre-commit or use a separate cache '
            f'directory (PRE_COMMIT_HOME).',
        )

    @contextlib.contextmanager
    def _transaction(
        self,
    ) -> Generator[sqlite3.Connection]:
        """An explicit, atomic write transaction.

        the driver's implicit transaction handling is disabled so that the
        schema changes together with the version bump either all commit or
        are all rolled back.
        """
        with self.connect() as db:
            db.isolation_level = None  # autocommit: we drive the transaction
            db.execute('BEGIN IMMEDIATE')
            try:
                yield db
            except BaseException:
                db.execute('ROLLBACK')
                raise
            else:
                db.execute('COMMIT')

    def _reconcile_schema(self) -> None:
        """Bring the database to the schema version understood here.

        * current version -- nothing to do
        * older version   -- pending migrations are applied in order; the
                             existing repositories and configuration records
                             are preserved
        * newer version   -- the database is left untouched and a fatal
                             error is raised

        the upgrade runs inside a single transaction, so a failure never
        leaves a partially modified schema behind.
        """
        with self.connect() as db:
            version = self._get_user_version(db)

        if version == SCHEMA_VERSION:
            return
        elif version > SCHEMA_VERSION:
            raise self._newer_version_error(version)
        elif self.readonly:  # pragma: win32 no cover
            # cannot upgrade a read-only cache -- leave it untouched
            return

        logger.info(
            f'upgrading pre-commit cache database schema '
            f'{version} -> {SCHEMA_VERSION}',
        )
        with self.exclusive_lock():
            # re-check under the lock: another process may have migrated
            with self._transaction() as db:
                version = self._get_user_version(db)
                if version == SCHEMA_VERSION:  # pragma: no cover (race)
                    return
                elif version > SCHEMA_VERSION:  # pragma: no cover (race)
                    raise self._newer_version_error(version)

                for to_version in range(version + 1, SCHEMA_VERSION + 1):
                    _MIGRATIONS[to_version](db)
                db.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')

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

    @staticmethod
    def _create_configs_table(db: sqlite3.Connection) -> None:
        db.execute(
            'CREATE TABLE IF NOT EXISTS configs ('
            '   path TEXT NOT NULL,'
            '   PRIMARY KEY (path)'
            ')',
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


# database upgrades, keyed by the schema version they upgrade to.
# version 0 databases were created by releases which did not record a
# version: they always contain the `repos` table but pre-date the `configs`
# table, which the version 1 upgrade creates when it is missing.
_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    1: Store._create_configs_table,
}
