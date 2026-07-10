"""Persistent application settings (shared across all projects).

Currently holds the common ``projects_root`` — the folder under which projects
are created (set from the GUI's Settings page). Stored as JSON in the data dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import settings


# The shared roots the first-run setup notice nags about.
SETUP_KEYS = ("projects_root", "ssd_scratch_root")
_SILENCED_KEY = "setup_notice_silenced"


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
    # A root that now has a value loses its "silenced" mark, so if it is ever
    # cleared again the setup notice comes back instead of staying hidden.
    silenced = current.get(_SILENCED_KEY)
    if isinstance(silenced, list):
        current[_SILENCED_KEY] = [k for k in silenced if not current.get(k)]
    _path().write_text(json.dumps(current, indent=2))
    return current


def setup_status() -> dict[str, Any]:
    """Which shared roots still need the user's attention.

    ``unset``   — never configured. These are what "close forever" silences.
    ``missing`` — configured, but the folder is gone from disk. Always surfaced,
                  silenced or not: a vanished root breaks every run pointing at it.
    """
    data = load()
    raw_silenced = data.get(_SILENCED_KEY)
    silenced = [k for k in raw_silenced if k in SETUP_KEYS] if isinstance(raw_silenced, list) else []

    unset: list[str] = []
    missing: list[str] = []
    values: dict[str, str] = {}
    for key in SETUP_KEYS:
        raw = data.get(key)
        if not raw:
            unset.append(key)
        elif not Path(raw).expanduser().is_dir():
            missing.append(key)
            values[key] = str(raw)

    return {
        "unset": unset,
        "missing": missing,
        "values": values,
        "silenced": silenced,
        "show": bool(missing) or any(k not in silenced for k in unset),
    }


def silence_setup_notice() -> dict[str, Any]:
    """Permanently silence the notice for the roots that are currently unset."""
    status = setup_status()
    update({_SILENCED_KEY: sorted(set(status["unset"]) | set(status["silenced"]))})
    return setup_status()


def projects_root() -> Path | None:
    raw = load().get("projects_root")
    return Path(raw).expanduser() if raw else None


def ssd_scratch_root() -> Path | None:
    raw = load().get("ssd_scratch_root")
    return Path(raw).expanduser() if raw else None


def source_repos() -> dict[str, str]:
    """User-configured source-clone paths per package, for the branch panel to
    manage a dev checkout that the install itself doesn't point at (e.g. a PyPI
    install plus a separate git clone)."""
    raw = load().get("source_repos")
    return dict(raw) if isinstance(raw, dict) else {}


def set_source_repo(name: str, path: str | None) -> dict[str, str]:
    """Set (or, with a falsy ``path``, clear) the source clone for ``name``."""
    repos = source_repos()
    if path:
        repos[name] = path
    else:
        repos.pop(name, None)
    update({"source_repos": repos})
    return repos
