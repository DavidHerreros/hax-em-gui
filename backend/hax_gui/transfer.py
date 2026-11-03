"""Project export/import: move a project (folder + job records) between machines.

The export is a plain ``.tar.gz`` of the project folder plus a manifest
(``hax_gui_export.json``) carrying the job rows, so the receiving server can
rebuild its database entries — runs keep their logs, outputs, bindings and
provenance. Paths inside the rows are rewritten from the source project folder
to the destination one; paths outside the project (raw data on the old machine)
are left as they are, which is honest: the data either exists at the same path
or the run must be re-pointed anyway.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import projects
from .jobs import ACTIVE, manager

MANIFEST = "hax_gui_export.json"
FORMAT = 1


class TransferError(Exception):
    pass


def export_project(project_id: str) -> Path:
    """Build the archive in a temp file and return its path (caller deletes)."""
    proj = projects.get(project_id)
    if proj is None:
        raise TransferError(f"unknown project '{project_id}'")
    rows = manager.list_for_project(project_id)
    manifest = {
        "format": FORMAT,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project": proj,
        "jobs": rows,
    }
    out = Path(tempfile.mkstemp(suffix=".tar.gz", prefix=f"hax_export_{project_id}_")[1])
    with tarfile.open(out, "w:gz") as tar:
        tar.add(proj["path"], arcname=project_id)
        mf = json.dumps(manifest, indent=2, default=str).encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(mf)
        import io
        tar.addfile(info, io.BytesIO(mf))
    return out


def _safe_members(tar: tarfile.TarFile, root: str) -> list[tarfile.TarInfo]:
    """Only plain files/dirs whose path stays inside the archive root — a
    crafted archive must not write outside the projects folder."""
    picked = []
    for m in tar.getmembers():
        if m.name == MANIFEST:
            continue
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != root:
            continue
        if not (m.isfile() or m.isdir()):
            continue          # no symlinks/devices from an untrusted archive
        picked.append(m)
    return picked


def import_project(archive: Path) -> dict[str, Any]:
    """Unpack an export into the projects root and re-create its job rows.

    The project folder gets a fresh name if the original is taken. Job rows keep
    their ids; a row whose id already exists on this server is skipped (that is
    the re-import-onto-the-source case, where the run is already here). Rows in
    an active state are imported as 'interrupted' — their process lives on the
    exporting machine, not this one.
    """
    root = projects._root()   # raises ProjectsRootUnset — surfaced by the route
    try:
        tar = tarfile.open(archive, "r:gz")
    except tarfile.TarError as exc:
        raise TransferError(f"not a project export: {exc}")
    with tar:
        try:
            manifest = json.loads(tar.extractfile(MANIFEST).read())
        except (KeyError, AttributeError, json.JSONDecodeError):
            raise TransferError("archive has no readable hax_gui_export.json manifest")
        if manifest.get("format") != FORMAT:
            raise TransferError(f"unsupported export format {manifest.get('format')}")
        src = manifest["project"]
        old_id, old_path = src["id"], src.get("path", "")

        new_id = old_id
        i = 1
        while (root / new_id).exists():
            i += 1
            new_id = f"{old_id}_{i}"
        members = _safe_members(tar, old_id)
        if not members:
            raise TransferError("archive holds no files under its project folder")
        tmp_root = Path(tempfile.mkdtemp(prefix="hax_import_", dir=root))
        try:
            tar.extractall(tmp_root, members=members)
            shutil.move(str(tmp_root / old_id), str(root / new_id))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    new_dir = root / new_id
    # The folder name IS the project id: rewrite the metadata to match.
    from . import jsonfile
    meta = jsonfile.read(new_dir / "hax_project.json", {}) or {}
    meta["id"] = new_id
    meta.setdefault("name", src.get("name", new_id))
    jsonfile.write(new_dir / "hax_project.json", meta)

    imported, skipped = [], []
    for row in reversed(manifest.get("jobs", [])):      # oldest first, keeps list order
        if manager.get(row["id"]) is not None:
            skipped.append(row["id"])
            continue
        def _remap(value: str | None) -> str | None:
            if not value or not old_path:
                return value
            return value.replace(old_path, str(new_dir))
        status = row.get("status", "interrupted")
        if status in ACTIVE:
            status = "interrupted"     # its process lives on the exporting machine
        command = [(_remap(c) if isinstance(c, str) else c) for c in (row.get("command") or [])]
        manager._exec(
            "INSERT INTO jobs (id, program, name, args, gpu, command, status, return_code, "
            "log_path, created_at, started_at, finished_at, project_id, workdir, cwd, bindings, provenance) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["id"], row["program"], row.get("name"),
             json.dumps({k: (_remap(v) if isinstance(v, str) else v)
                         for k, v in (row.get("args") or {}).items()}),
             row.get("gpu"), json.dumps(command), status, row.get("return_code"),
             _remap(row.get("log_path")) or "", row.get("created_at"),
             row.get("started_at"), row.get("finished_at"), new_id,
             _remap(row.get("workdir")), _remap(row.get("cwd")),
             json.dumps(row["bindings"]) if row.get("bindings") else None,
             json.dumps(row["provenance"]) if row.get("provenance") else None),
        )
        imported.append(row["id"])
    return {"project": projects.get(new_id), "imported": len(imported), "skipped": skipped}
