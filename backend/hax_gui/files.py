"""Server-side file browser for path-typed arguments.

By default the browser is **unrestricted**: it lists anything the server process can
read, so shared storage (``/data/<you>/...``) and other mounts work with no setup.
The browser is not a security boundary and never was — path-typed job arguments are
never confined (see ``command.py``), so confining the browser protects nothing while
making legitimate paths unreachable.

Setting ``file_roots`` opts in to confinement, for hygiene or for a shared box.
Then ``..`` cannot climb above a root: it is collapsed lexically before the check.
Containment is judged on the path the user *navigated*, not on where it finally
points, so a symlink placed inside a root (``~/scratch -> /mnt/fast/scratch``) is
followed like any other folder, as a file manager would.

``browse_home`` is where the picker opens, and is independent of the roots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import settings


class AccessDenied(PermissionError):
    pass


class InvalidName(ValueError):
    pass


def _norm(path: str | Path) -> Path:
    """Absolute and lexically normalized (``..``/``.`` collapsed), symlinks intact."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _roots() -> list[Path]:
    return [_norm(r) for r in settings.file_roots]


def _contains(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _within_roots(target: Path) -> bool:
    """True when ``target`` (already lexically normalized) is reachable.

    With no roots configured the browser is unrestricted, so everything is. Otherwise
    containment is compared as navigated, so symlinks under a root stay under it. The
    resolved forms are compared too, which lets a root that is *itself* a symlink
    still match requests naming its real location.
    """
    roots = _roots()
    if not roots:
        return True
    if any(_contains(root, target) for root in roots):
        return True
    resolved = target.resolve()
    return any(_contains(root.resolve(), resolved) for root in roots)


def _default_dir() -> Path:
    """Where a listing starts when no path is given."""
    home = _norm(settings.browse_home)
    if _within_roots(home):
        return home
    return _roots()[0]   # a home outside the configured roots would be unlistable


def resolve(path: str | None) -> Path:
    """Normalize a request path, and reject it if the roots put it out of reach."""
    target = _default_dir() if not path else _norm(path)
    if not _within_roots(target):
        raise AccessDenied(f"path outside allowed roots: {target}")
    return target


def _start_dir(path: str | None, fallback: bool) -> Path:
    """Locate the directory a listing should start from.

    With ``fallback`` (used when opening the picker at a path a form field already
    holds, which may be an output that does not exist yet, or lie out of reach) an
    unusable path degrades to the nearest existing reachable ancestor, or to the
    default directory — rather than erroring. Plain navigation keeps erroring, so a
    stale entry reports itself instead of silently showing somewhere else.
    """
    try:
        target = resolve(path)
    except AccessDenied:
        if not fallback:
            raise
        return _default_dir()

    if target.exists():
        return target
    if not fallback:
        raise FileNotFoundError(str(target))

    for parent in target.parents:
        if not _within_roots(parent):
            break
        if parent.is_dir():
            return parent
    return _default_dir()


def list_dir(path: str | None, fallback: bool = False) -> dict[str, Any]:
    target = _start_dir(path, fallback)

    # Naming a file lists its folder, and tells the client which entry was meant so
    # it can scroll to it — this is what makes pasting a copied file path useful.
    named_file = str(target) if target.is_file() else None
    if named_file:
        target = target.parent

    entries: list[dict[str, Any]] = []
    try:
        children = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError as exc:
        raise AccessDenied(str(exc)) from exc

    for child in children:
        try:
            is_dir = child.is_dir()
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": is_dir,
                    "hidden": child.name.startswith("."),   # dotfile; hidden by default in the UI
                    "size": None if is_dir else child.stat().st_size,
                }
            )
        except OSError:
            continue

    # "Up" stops at a configured root, and at "/" when unrestricted (whose parent is
    # itself, so it would otherwise never terminate).
    parent = target.parent
    if parent == target or not _within_roots(parent) or target in _roots():
        parent = None

    return {
        "path": str(target),
        "parent": str(parent) if parent else None,
        "roots": [str(r) for r in _roots()],
        "file": named_file,
        "entries": entries,
    }


def make_dir(path: str | None, name: str) -> dict[str, Any]:
    """Create one sub-directory `name` under `path`, and return the fresh listing."""
    name = name.strip()
    # Exactly one path component: rejects "", ".", "..", "a/b", "/abs" — anything
    # that could climb out of `path`. Note Path("..").name is ".." (not ""), so the
    # dot names need calling out explicitly.
    if not name or name in {".", ".."} or Path(name).name != name:
        raise InvalidName(f"not a valid directory name: {name!r}")

    base = resolve(path)
    if not base.exists():
        raise FileNotFoundError(str(base))
    if not base.is_dir():
        raise NotADirectoryError(str(base))

    target = base / name
    if not _within_roots(target):
        raise AccessDenied(f"path outside allowed roots: {target}")
    try:
        target.mkdir()   # non-recursive, and errors if the name is already taken
    except PermissionError as exc:
        raise AccessDenied(str(exc)) from exc

    return {"created": str(target), **list_dir(str(base))}
