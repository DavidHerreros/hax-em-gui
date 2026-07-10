"""Sandboxed server-side file browser for path-typed arguments.

Listings are confined to the configured ``file_roots``: the SPA can never climb
above them with ``..``, which is collapsed lexically before the check.

Containment is judged on the path the user *navigated*, not on where it finally
points, so a symlink placed inside a root (``~/scratch -> /mnt/fast/scratch``) is
followed like any other folder — as a file manager would. Note this is a usability
guardrail, not a security boundary: path-typed job arguments are never confined to
the roots (see ``command.py``), so the browser is not the thing keeping anyone out.
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
    """True when ``target`` (already lexically normalized) sits under a root.

    Compared as navigated, so symlinks under a root stay under it. The resolved
    forms are compared too, which lets a root that is *itself* a symlink still
    match requests naming its real location.
    """
    roots = _roots()
    if any(_contains(root, target) for root in roots):
        return True
    resolved = target.resolve()
    return any(_contains(root.resolve(), resolved) for root in roots)


def resolve(path: str | None) -> Path:
    """Normalize a request path, defaulting to (and confined within) the roots."""
    target = _roots()[0] if not path else _norm(path)
    if not _within_roots(target):
        raise AccessDenied(f"path outside allowed roots: {target}")
    return target


def _start_dir(path: str | None, fallback: bool) -> Path:
    """Locate the directory a listing should start from.

    With ``fallback`` (used when opening the picker at a path a form field already
    holds, which may be an output that does not exist yet, or lie outside the roots)
    an unusable path degrades to the nearest existing ancestor inside a root, or to
    the first root — rather than erroring. Plain navigation keeps erroring, so a
    stale entry reports itself instead of silently showing somewhere else.
    """
    try:
        target = resolve(path)
    except AccessDenied:
        if not fallback:
            raise
        return _roots()[0]

    if target.exists():
        return target
    if not fallback:
        raise FileNotFoundError(str(target))

    for parent in target.parents:
        if not _within_roots(parent):
            break
        if parent.is_dir():
            return parent
    return _roots()[0]


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

    parent = target.parent if _within_roots(target.parent) and target not in _roots() else None
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
