"""Project management — CryoSPARC-style.

A project is a folder under the common ``projects_root`` containing a
``hax_project.json`` metadata file. Each program run inside a project gets its own
subfolder — named after the job (``<program>_<job_id>``, so re-running keeps the path;
see :meth:`jobs.JobManager.create`) — holding all of its inputs-by-reference, outputs
and logs, so users never have to specify an output path.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import settings_store

META = "hax_project.json"


class ProjectsRootUnset(Exception):
    """Raised when no projects_root has been configured in Settings."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root() -> Path:
    root = settings_store.projects_root()
    if root is None:
        raise ProjectsRootUnset("No projects folder configured. Set one in Settings.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")
    return s or "project"


def _is_direct_child(folder: Path, root: Path) -> bool:
    try:
        return folder.resolve().parent == root.resolve()
    except OSError:
        return False


def list_projects() -> list[dict[str, Any]]:
    root = _root()
    out = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        meta = child / META
        if child.is_dir() and meta.exists():
            try:
                out.append(_public(json.loads(meta.read_text()), child))
            except (json.JSONDecodeError, OSError):
                continue
    return out


def create(name: str) -> dict[str, Any]:
    root = _root()
    slug = _slug(name)
    folder = root / slug
    i = 1
    while folder.exists():
        i += 1
        folder = root / f"{slug}_{i}"
    folder.mkdir(parents=True)
    data = {"id": folder.name, "name": name.strip() or folder.name, "created_at": _now()}
    (folder / META).write_text(json.dumps(data, indent=2))
    return _public(data, folder)


def get(project_id: str) -> dict[str, Any] | None:
    folder = dir_for(project_id)
    if folder is None:
        return None
    return _public(json.loads((folder / META).read_text()), folder)


def dir_for(project_id: str) -> Path | None:
    root = _root()
    folder = root / project_id
    if not _is_direct_child(folder, root) or not (folder / META).exists():
        return None
    return folder


def rename(project_id: str, new_name: str) -> dict[str, Any] | None:
    """Change a project's display name. The folder (and thus the id that jobs
    reference) is left untouched, so existing runs keep resolving."""
    folder = dir_for(project_id)
    if folder is None:
        return None
    meta = folder / META
    data = json.loads(meta.read_text())
    data["name"] = new_name.strip() or data.get("name", folder.name)
    meta.write_text(json.dumps(data, indent=2))
    return _public(data, folder)


def delete(project_id: str) -> bool:
    """Remove a project folder and everything inside it (all of its runs)."""
    folder = dir_for(project_id)
    if folder is None:
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


def _public(data: dict[str, Any], folder: Path) -> dict[str, Any]:
    return {
        "id": data.get("id", folder.name),
        "name": data.get("name", folder.name),
        "created_at": data.get("created_at"),
        "path": str(folder),
    }
