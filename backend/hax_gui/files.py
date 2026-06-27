"""Sandboxed server-side file browser for path-typed arguments.

Listings are confined to the configured ``file_roots`` so the SPA can never
escape above them via ``..`` or symlinks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import settings


class AccessDenied(PermissionError):
    pass


def _roots() -> list[Path]:
    return [Path(r).expanduser().resolve() for r in settings.file_roots]


def _within_roots(target: Path) -> bool:
    return any(target == root or root in target.parents for root in _roots())


def resolve(path: str | None) -> Path:
    """Resolve a request path, defaulting to (and confined within) the roots."""
    roots = _roots()
    target = roots[0] if not path else Path(path).expanduser()
    target = target.resolve()
    if not _within_roots(target):
        raise AccessDenied(f"path outside allowed roots: {target}")
    return target


def list_dir(path: str | None) -> dict[str, Any]:
    target = resolve(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    if target.is_file():
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
        "entries": entries,
    }
