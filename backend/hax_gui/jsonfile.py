"""Atomic, self-recovering JSON files (settings.json, hax_project.json).

These files used to be rewritten with a bare ``write_text`` — a crash or full
disk mid-write left them truncated, and a truncated ``settings.json`` silently
made every project vanish (its ``projects_root`` unreadable). Writes now go
through temp-file + ``os.replace`` with the previous good version kept as
``.bak``; a corrupt file is quarantined (``.corrupt``, so it can be inspected)
and the ``.bak`` restored instead of the damage being swallowed as "empty".
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger("hax_gui")


def write(path: Path, data: Any) -> None:
    """Atomically replace ``path`` with ``data`` (previous version kept as .bak)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    if path.exists():
        with contextlib.suppress(OSError):
            shutil.copy2(path, path.with_name(path.name + ".bak"))
    os.replace(tmp, path)


def read(path: Path, default: Any = None) -> Any:
    """Read JSON from ``path``; a missing file returns ``default``.

    A *corrupt* file is moved aside to ``.corrupt`` and the ``.bak`` written by
    :func:`write` is restored and returned — so one interrupted write costs at
    most the last update, never the whole file.
    """
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except OSError:
        return default
    except json.JSONDecodeError:
        pass
    quarantine = path.with_name(path.name + ".corrupt")
    with contextlib.suppress(OSError):
        os.replace(path, quarantine)
    bak = path.with_name(path.name + ".bak")
    try:
        data = json.loads(bak.read_text())
    except (OSError, json.JSONDecodeError):
        log.warning("%s is corrupt (quarantined as %s) and no backup was usable",
                    path, quarantine.name)
        return default
    log.warning("%s was corrupt (quarantined as %s); restored the previous version",
                path, quarantine.name)
    with contextlib.suppress(OSError):
        shutil.copy2(bak, path)
    return data
