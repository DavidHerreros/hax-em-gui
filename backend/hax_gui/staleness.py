"""Detect on-disk edits to the hax / hax_gui source trees (editable installs).

Two staleness problems this backs: a reused viewer subprocess keeps whatever
module it imported at spawn (see viewer.py / mrc_viewer.py / md_viewer.py), and
the cached program-schemas file (registry.py) can outlive the argparse it was
extracted from. Both need "has anything changed on disk since X" cheaply and
often, hence the short cache below.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .registry import _hax_package_dir

_CACHE_TTL = 5.0
_cache: tuple[float, float] | None = None   # (stamp, value)


def _newest_py_mtime(root: Path | None) -> float:
    if root is None or not root.is_dir():
        return 0.0
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            try:
                mtime = os.path.getmtime(os.path.join(dirpath, name))
            except OSError:
                continue
            if mtime > newest:
                newest = mtime
    return newest


def sources_mtime() -> float:
    """Newest ``.py`` mtime across the installed ``hax`` and ``hax_gui`` trees.

    0.0 when neither tree can be located. Cached for a few seconds so a burst of
    viewer ``ensure()`` calls doesn't each walk both trees.
    """
    global _cache
    now = time.time()
    if _cache is not None and now - _cache[0] < _CACHE_TTL:
        return _cache[1]
    hax_gui_dir = Path(__file__).resolve().parent
    value = max(_newest_py_mtime(_hax_package_dir()), _newest_py_mtime(hax_gui_dir))
    _cache = (now, value)
    return value


_hax_cache: tuple[float, float] | None = None   # (stamp, value)


def hax_sources_mtime() -> float:
    """Newest ``.py`` mtime across just the ``hax`` tree (not ``hax_gui``).

    Used for the schema-cache staleness check (2b): a ``hax_gui`` edit doesn't
    change what the program schemas extracted from ``hax``'s argparse look like,
    so that check only needs to watch ``hax`` itself. Cached separately from
    :func:`sources_mtime` since the two are polled on different schedules.
    """
    global _hax_cache
    now = time.time()
    if _hax_cache is not None and now - _hax_cache[0] < _CACHE_TTL:
        return _hax_cache[1]
    value = _newest_py_mtime(_hax_package_dir())
    _hax_cache = (now, value)
    return value
