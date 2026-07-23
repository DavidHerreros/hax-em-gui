"""How much disk a project or a run occupies.

Sizes are measured by walking the folder with ``os.scandir`` — no ``du`` subprocess,
so it behaves the same wherever the server runs — and they are *apparent* sizes
(``st_size``), which is what a file listing shows. On a filesystem with compression
or sparse files the blocks actually consumed can differ.

A walk is metadata-only and cheap per file, but a run folder can hold an
array-record cache of a whole particle stack, and on network storage even stat()
costs. Results are therefore cached for :data:`TTL` seconds and every measurement is
meant to be run off the event loop (``asyncio.to_thread``). Symlinked directories are
never descended into: they usually point outside the folder being measured, and
counting their contents would both double-count and risk a loop.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

# How long a measurement is reused before the folder is walked again. Long enough
# that opening a project, switching tabs and coming back doesn't re-walk it; short
# enough that a finished run shows up as growth without a manual refresh.
TTL = 120.0

_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_lock = threading.Lock()


def measure(path: str | Path | None, refresh: bool = False) -> dict[str, Any] | None:
    """Total bytes and file count under ``path`` (None when there is no path).

    Blocking: call it from a worker thread. Repeated calls inside :data:`TTL` return
    the previous result, unless ``refresh`` forces a fresh walk.
    """
    if not path:
        return None
    key = str(path)
    if not refresh:
        hit = cached(key)
        if hit is not None:
            return hit
    folder = Path(key)
    if not folder.is_dir():
        # Recorded as a real (empty) measurement: a run whose folder was removed, or
        # never created, is a legitimate "0 bytes" rather than an error.
        result = {"path": key, "bytes": 0, "files": 0, "exists": False, "measured_at": time.time()}
    else:
        total, files = _walk(folder)
        result = {"path": key, "bytes": total, "files": files, "exists": True,
                  "measured_at": time.time()}
    with _lock:
        _cache[key] = (result["measured_at"], result)
    return result


def cached(path: str | Path | None) -> dict[str, Any] | None:
    """The last measurement of ``path`` if it is still fresh, else None."""
    if not path:
        return None
    with _lock:
        hit = _cache.get(str(path))
    if hit is None or time.time() - hit[0] > TTL:
        return None
    return hit[1]


def forget(path: str | Path | None) -> None:
    """Drop a cached measurement — for a folder that has just been deleted or rerun."""
    if not path:
        return
    with _lock:
        _cache.pop(str(path), None)


def _walk(folder: Path) -> tuple[int, int]:
    """(bytes, files) under ``folder``. Unreadable entries are skipped, not raised:
    a partial total is far more useful here than an error."""
    total = files = 0
    stack = [str(folder)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue          # vanished mid-walk, or not ours to stat
        except OSError:
            continue                      # unreadable directory
    return total, files
