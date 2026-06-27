"""Persistent application settings (shared across all projects).

Currently holds the common ``projects_root`` — the folder under which projects
are created (set from the GUI's Settings page). Stored as JSON in the data dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import settings


def _path() -> Path:
    return settings.data_dir / "settings.json"


def load() -> dict[str, Any]:
    p = _path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def update(data: dict[str, Any]) -> dict[str, Any]:
    settings.ensure_dirs()
    current = load()
    for k, v in data.items():
        if v is not None:
            current[k] = v
    _path().write_text(json.dumps(current, indent=2))
    return current


def projects_root() -> Path | None:
    raw = load().get("projects_root")
    return Path(raw).expanduser() if raw else None


def ssd_scratch_root() -> Path | None:
    raw = load().get("ssd_scratch_root")
    return Path(raw).expanduser() if raw else None
