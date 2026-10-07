from __future__ import annotations

import contextlib
import logging
import os
import os.path
import shlex
import stat
import sys
import tempfile
from typing import NamedTuple

from pre_commit import git
from pre_commit import output
from pre_commit.clientlib import InvalidConfigError
from pre_commit.clientlib import load_config
from pre_commit.repository import all_hooks
from pre_commit.repository import install_hook_envs
from pre_commit.store import Store
from pre_commit.util import resource_text


logger = logging.getLogger(__name__)

# This is used to identify the hook file we install
PRIOR_HASHES = (
    b'4d9958c90bc262f47553e2c073f14cfe',
    b'd8ee923c46731b42cd95cc869add4062',
    b'49fd668cb42069aa1b6048464be5d395',
    b'79f09a650522a87b0da915d0d983b2de',
    b'e358c9dae00eac5d06b38dfdb1e33a8c',
)
CURRENT_HASH = b'138fd403232d2ddd5efb44317e38bf03'
TEMPLATE_START = '# start templated\n'
TEMPLATE_END = '# end templated\n'


def _hook_types(cfg_filename: str, hook_types: list[str] | None) -> list[str]:
    if hook_types is not None:
        return hook_types
    else:
        try:
            cfg = load_config(cfg_filename)
        except InvalidConfigError:
            return ['pre-commit']
        else:
            return cfg['default_install_hook_types']


def _hook_paths(
        hook_type: str,
        git_dir: str | None = None,
) -> tuple[str, str]:
    git_dir = git_dir if git_dir is not None else git.get_git_common_dir()
    pth = os.path.join(git_dir, 'hooks', hook_type)
    return pth, f'{pth}.legacy'


def is_our_script(filename: str) -> bool:
    if not os.path.exists(filename):  # pragma: win32 no cover (symlink)
        return False
    with open(filename, 'rb') as f:
        contents = f.read()
    return any(h in contents for h in (CURRENT_HASH,) + PRIOR_HASHES)


def _render_hook(
        existing: str | None,
        config_file: str,
        hook_type: str,
        skip_on_missing_config: bool,
) -> str:
    """Render the desired hook script.

    Only the section between ``TEMPLATE_START`` and ``TEMPLATE_END`` is
    maintained by pre-commit -- anything an existing installation contains
    outside of these markers is preserved verbatim.
    """
    contents = resource_text('hook-tmpl')
    before, rest = contents.split(TEMPLATE_START)
    _, after = rest.split(TEMPLATE_END)

    # on windows always use `/bin/sh` since `bash` might not be on PATH
    # though we use bash-specific features `sh` on windows is actually
    # bash in "POSIXLY_CORRECT" mode which still supports the features we
    # use: subshells / arrays
    windows_shebang = sys.platform == 'win32'  # pragma: win32 cover

    if existing is not None:
        start = existing.find(TEMPLATE_START)
        end = existing.find(TEMPLATE_END)
        if 0 <= start < end:
            # keep the user's portions of the script untouched
            before = existing[:start]
            after = existing[end + len(TEMPLATE_END):]
            windows_shebang = False

    args = ['hook-impl', f'--config={config_file}', f'--hook-type={hook_type}']
    if skip_on_missing_config:
        args.append('--skip-on-missing-config')

    ret = ''
    if windows_shebang:  # pragma: win32 cover
        ret += '#!/bin/sh\n'
    ret += before
    ret += TEMPLATE_START
    ret += f'INSTALL_PYTHON={shlex.quote(sys.executable)}\n'
    ret += f'ARGS=({shlex.join(args)})\n'
    ret += TEMPLATE_END
    ret += after
    return ret


class _PendingHook(NamedTuple):
    hook_path: str
    legacy_path: str
    contents: str
    unchanged: bool
    # an existing non-pre-commit hook is moved to the legacy path
    migrate_legacy: bool
    # the legacy hook is removed on install (-f)
    overwrite: bool
    mode: int


def _prepare_hook_install(
        config_file: str,
        hook_type: str,
        overwrite: bool,
        skip_on_missing_config: bool,
        git_dir: str | None,
        created_dirs: list[str],
) -> _PendingHook:
    hook_path, legacy_path = _hook_paths(hook_type, git_dir=git_dir)

    hooks_dir = os.path.dirname(hook_path)
    if not os.path.isdir(hooks_dir):
        os.makedirs(hooks_dir, exist_ok=True)
        created_dirs.append(hooks_dir)

    existing: str | None = None
    migrate_legacy = False
    mode = 0o755
    if os.path.lexists(hook_path):
        if is_our_script(hook_path):
            with open(hook_path) as f:
                existing = f.read()
            mode = stat.S_IMODE(os.stat(hook_path).st_mode)
        else:
            migrate_legacy = True

    contents = _render_hook(
        existing, config_file, hook_type, skip_on_missing_config,
    )

    return _PendingHook(
        hook_path=hook_path,
        legacy_path=legacy_path,
        contents=contents,
        unchanged=existing == contents,
        migrate_legacy=migrate_legacy,
        overwrite=overwrite,
        mode=mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )


def _tmp_sibling(path: str) -> str:
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f'.{os.path.basename(path)}.',
        suffix='.tmp',
    )
    os.close(fd)
    return tmp_path


def _stage_contents(path: str, contents: str, mode: int) -> str:
    """Write ``contents`` next to ``path`` without touching ``path``.

    The file is fully written, fsynced and made executable before it is
    moved into place by the caller, so an interrupted install can never
    leave a half-written hook script behind.
    """
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f'.{os.path.basename(path)}.',
        suffix='.tmp',
    )
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, mode)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise
    return tmp_path


def _install_prepared(pending: list[_PendingHook]) -> None:
    """Install all prepared hooks, all or nothing.

    New contents are staged first and existing files are only displaced by
    atomic renames.  If anything fails for any hook every prior rename is
    reversed, leaving all hook scripts in their original state.
    """
    # (src, dst) renames which reverse a rename already performed
    undos: list[tuple[str, str]] = []
    leftovers: list[str] = []

    def _replace(src: str, dst: str) -> None:
        os.replace(src, dst)
        undos.append((dst, src))

    try:
        # stage every new script before modifying any existing one
        staged: dict[str, str] = {}
        for p in pending:
            if not p.unchanged:
                tmp_path = _stage_contents(p.hook_path, p.contents, p.mode)
                staged[p.hook_path] = tmp_path
                leftovers.append(tmp_path)

        for p in pending:
            if p.unchanged:
                # the script is already in the desired state -- don't touch
                # the file so its mtime / inode / permissions are preserved
                output.write_line(f'pre-commit installed at {p.hook_path}')
                continue

            hook_path = p.hook_path
            legacy_path = p.legacy_path

            # the existing legacy hook is either replaced by the migrated
            # hook or removed when overwriting -- park it so the move can be
            # rolled back
            if (
                    os.path.lexists(legacy_path) and
                    (p.migrate_legacy or p.overwrite)
            ):
                legacy_backup = _tmp_sibling(legacy_path)
                leftovers.append(legacy_backup)
                _replace(legacy_path, legacy_backup)

            if p.migrate_legacy:
                # move the existing non-pre-commit hook out of the way
                _replace(hook_path, legacy_path)
            elif os.path.lexists(hook_path):
                # park our existing script so its bytes survive a rollback
                hook_backup = _tmp_sibling(hook_path)
                leftovers.append(hook_backup)
                _replace(hook_path, hook_backup)

            _replace(staged[hook_path], hook_path)

            # with --overwrite the migrated hook is discarded -- it is
            # parked rather than deleted so a rollback can still restore it
            if p.overwrite and os.path.lexists(legacy_path):
                legacy_trash = _tmp_sibling(legacy_path)
                leftovers.append(legacy_trash)
                _replace(legacy_path, legacy_trash)

            output.write_line(f'pre-commit installed at {hook_path}')

            if not p.overwrite and os.path.lexists(legacy_path):
                output.write_line(
                    f'Running in migration mode with existing hooks at '
                    f'{legacy_path}\n'
                    f'Use -f to use only pre-commit.',
                )
    except BaseException:
        for src, dst in reversed(undos):
            with contextlib.suppress(OSError):
                if os.path.lexists(src):
                    os.replace(src, dst)
        for leftover in leftovers:
            with contextlib.suppress(FileNotFoundError):
                os.remove(leftover)
        raise
    else:
        for leftover in leftovers:
            with contextlib.suppress(FileNotFoundError):
                os.remove(leftover)


def install(
        config_file: str,
        store: Store,
        hook_types: list[str] | None,
        overwrite: bool = False,
        hooks: bool = False,
        skip_on_missing_config: bool = False,
        git_dir: str | None = None,
) -> int:
    if git_dir is None and git.has_core_hookpaths_set():
        logger.error(
            'Cowardly refusing to install hooks with `core.hooksPath` set.\n'
            'hint: `git config --unset-all core.hooksPath`',
        )
        return 1

    created_dirs: list[str] = []
    try:
        pending = [
            _prepare_hook_install(
                config_file, hook_type,
                overwrite=overwrite,
                skip_on_missing_config=skip_on_missing_config,
                git_dir=git_dir,
                created_dirs=created_dirs,
            )
            for hook_type in _hook_types(config_file, hook_types)
        ]
    except BaseException:
        # only directories which did not previously exist are removed
        for directory in reversed(created_dirs):
            with contextlib.suppress(OSError):
                os.rmdir(directory)
        raise

    _install_prepared(pending)

    if hooks:
        install_hooks(config_file, store)

    return 0


def install_hooks(config_file: str, store: Store) -> int:
    install_hook_envs(all_hooks(load_config(config_file), store), store)
    return 0


def _uninstall_hook_script(hook_type: str) -> None:
    hook_path, legacy_path = _hook_paths(hook_type)

    # If our file doesn't exist or it isn't ours, gtfo.
    if not os.path.exists(hook_path) or not is_our_script(hook_path):
        return

    os.remove(hook_path)
    output.write_line(f'{hook_type} uninstalled')

    if os.path.exists(legacy_path):
        os.replace(legacy_path, hook_path)
        output.write_line(f'Restored previous hooks to {hook_path}')


def uninstall(config_file: str, hook_types: list[str] | None) -> int:
    for hook_type in _hook_types(config_file, hook_types):
        _uninstall_hook_script(hook_type)
    return 0
