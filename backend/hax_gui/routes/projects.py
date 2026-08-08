"""Project routes: CRUD, usage, per-project job list/creation, relink, sources."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from .. import adapters, projects, settings_store, usage
from ..command import ValidationError
from ..jobs import JobActive, ProjectNotFound, manager
from ..labels import manager as label_manager
from ..projects import ProjectsRootUnset
from ..workflows import manager as wf_manager
from .. import chats as chat_store
from .shared import (
    JobCreate, RerunBody, apply_cascade as _apply_cascade, attach_labels as _attach_labels,
    auth, project_or_404 as _project_or_404, queue_cascade as _queue_cascade,
)

router = APIRouter()

# -- projects --------------------------------------------------------------
class ProjectCreate(BaseModel):
    name: str


class ProjectRename(BaseModel):
    name: str


def _with_run_counts(projs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each project with the number of runs it holds (for the cards).

    One GROUP BY, not a full job-row fetch (args/command/bindings JSON-parsed)
    per project per 5-second poll just to call len() — and no longer capped at
    list_for_project's 200-row limit."""
    counts = manager.count_by_project()
    for proj in projs:
        proj["runs"] = counts.get(proj["id"], 0)
    return projs


@router.get("/api/projects", dependencies=[Depends(auth)])
def list_projects() -> list[dict[str, Any]]:
    try:
        return _with_run_counts(projects.list_projects())
    except ProjectsRootUnset:
        return []


@router.post("/api/projects", dependencies=[Depends(auth)])
def create_project(body: ProjectCreate) -> dict[str, Any]:
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="project name required")
    try:
        return projects.create(body.name)
    except ProjectsRootUnset as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/api/projects/usage", dependencies=[Depends(auth)])
async def projects_usage(refresh: bool = False) -> list[dict[str, Any]]:
    """Disk occupied by each project folder.

    Declared before /api/projects/{project_id} so that path doesn't swallow it. The
    walk is off the event loop and cached (see `usage`), so the repeat cost of the
    rail showing these is nothing; `refresh=1` forces a re-measure.
    """
    try:
        plist = projects.list_projects()
    except ProjectsRootUnset:
        return []

    def _all() -> list[dict[str, Any]]:
        out = []
        for proj in plist:
            measured = usage.measure(proj["path"], refresh) or {}
            out.append({"id": proj["id"], "name": proj["name"], "path": proj["path"],
                        "bytes": measured.get("bytes", 0), "files": measured.get("files", 0),
                        "measured_at": measured.get("measured_at")})
        return out

    return await asyncio.to_thread(_all)


@router.get("/api/projects/{project_id}", dependencies=[Depends(auth)])
def get_project(project_id: str) -> dict[str, Any]:
    proj = _project_or_404(project_id)
    return proj


@router.put("/api/projects/{project_id}", dependencies=[Depends(auth)])
def rename_project(project_id: str, body: ProjectRename) -> dict[str, Any]:
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="project name required")
    _project_or_404(project_id)
    proj = projects.rename(project_id, body.name)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"unknown project '{project_id}'")
    return proj


@router.delete("/api/projects/{project_id}", dependencies=[Depends(auth)])
def delete_project(project_id: str) -> dict[str, bool]:
    _project_or_404(project_id)
    job_ids = [j["id"] for j in manager.list_for_project(project_id)]
    try:
        manager.delete_for_project(project_id)
    except JobActive:
        raise HTTPException(status_code=409, detail="cancel the project's running jobs before deleting it")
    # Everything filed under the project goes with it: label associations,
    # workflows (rows + any live scheduler), and chats.
    label_manager.remove_jobs(job_ids)
    wf_manager.delete_for_project(project_id)
    chat_store.manager.delete_for_project(project_id)
    projects.delete(project_id)
    return {"deleted": True}


@router.get("/api/projects/{project_id}/jobs", dependencies=[Depends(auth)])
def project_jobs(project_id: str) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    return _attach_labels(manager.list_for_project(project_id))


@router.get("/api/projects/{project_id}/usage", dependencies=[Depends(auth)])
async def project_usage(project_id: str, refresh: bool = False) -> dict[str, Any]:
    """This project's disk use, broken down per run (largest first).

    `other_bytes` is what sits in the project folder outside its runs' folders: the
    project metadata, and anything a deleted job left behind. The SSD scratch is
    reported *beside* the total rather than inside it — it lives under the configured
    scratch root, not in the project, it can dwarf the project itself (it caches the
    whole particle stack), and it is rebuildable, so it is the first thing to reclaim.
    """
    proj = _project_or_404(project_id)
    project_jobs_list = manager.list_for_project(project_id)
    scratch_root = settings_store.ssd_scratch_root()
    scratch_dir = str(scratch_root / project_id) if scratch_root is not None else None

    def _measure() -> dict[str, Any]:
        total = usage.measure(proj["path"], refresh) or {}
        per_job, counted = [], 0
        for job in project_jobs_list:
            measured = usage.measure(job.get("workdir"), refresh)
            if measured is None:
                continue                        # saved job that never got a run folder
            counted += measured["bytes"]
            per_job.append({"id": job["id"], "name": job.get("name"), "program": job["program"],
                            "status": job["status"], "workdir": job.get("workdir"),
                            "bytes": measured["bytes"], "files": measured["files"]})
        per_job.sort(key=lambda j: j["bytes"], reverse=True)
        return {
            "id": proj["id"], "name": proj["name"], "path": proj["path"],
            "bytes": total.get("bytes", 0), "files": total.get("files", 0),
            "measured_at": total.get("measured_at"),
            "jobs": per_job,
            "other_bytes": max(0, total.get("bytes", 0) - counted),
            "scratch": usage.measure(scratch_dir, refresh),
        }

    return await asyncio.to_thread(_measure)


@router.post("/api/projects/{project_id}/relink", dependencies=[Depends(auth)])
def relink_project(project_id: str) -> dict[str, Any]:
    """Recover input bindings that an edit-and-rerun flattened into literal paths, so
    the workflow edges they back reappear. Returns what was reconnected."""
    _project_or_404(project_id)
    restored = manager.relink_project(project_id)
    return {"relinked": restored, "count": len(restored)}


@router.get("/api/projects/{project_id}/sources", dependencies=[Depends(auth)])
def project_sources(project_id: str, kind: str) -> list[dict[str, Any]]:
    """Jobs in this project whose output can feed a `kind` input (for chaining)."""
    _project_or_404(project_id)
    return adapters.compatible_sources(kind, manager.list_for_project(project_id))


@router.post("/api/projects/{project_id}/jobs", dependencies=[Depends(auth)])
async def create_project_job(project_id: str, body: JobCreate) -> dict[str, Any]:
    # Reuse an existing job in place only when it's a real, non-active job of this
    # project; otherwise fall back to creating a fresh one (e.g. it was deleted or
    # is already running), so a stale replace target can never error or clobber a run.
    replace = body.replace_job_id
    if replace:
        existing = manager.get(replace)
        if existing is None or existing.get("project_id") != project_id or existing.get("active"):
            replace = None
    # Editing a run in place re-makes its outputs exactly as a rerun does, so the same
    # invalidation applies to everything downstream. Done before the create call, which
    # deletes the folder those dependents read from — but only once the submission has
    # passed a dry validation: a 422 must not leave dependents killed/staled for a
    # rebuild that never happened.
    if replace:
        try:
            manager.precheck(body.program, body.args, body.gpu, body.bindings)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail={"errors": exc.errors})
    rerun = RerunBody(cascade=body.cascade, cancel_running=body.cancel_running)
    deps = await _apply_cascade([replace], rerun) if replace else []
    try:
        created = await manager.create(body.program, body.args, body.gpu, project_id, body.name, body.bindings, body.launch,
                                       replace_job_id=replace, recreate_ssd=body.recreate_ssd, resume=body.resume and bool(replace))
    except ProjectNotFound:
        raise HTTPException(status_code=404, detail=f"unknown project '{project_id}'")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})
    # Only a launched run can have dependents waiting on it; a merely-saved edit has
    # produced nothing for them to start from.
    if body.launch:
        await _queue_cascade(project_id, deps, rerun)
    return created




@router.get("/api/projects/{project_id}/export", dependencies=[Depends(auth)])
async def export_project(project_id: str):
    """Download the project as a .tar.gz (folder + job-record manifest), for
    moving it to another machine — see hax_gui/transfer.py."""
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    from .. import transfer

    _project_or_404(project_id)
    try:
        # Off the loop: tarring a project can walk many GB.
        path = await asyncio.to_thread(transfer.export_project, project_id)
    except transfer.TransferError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(str(path), media_type="application/gzip",
                        filename=f"{project_id}.hax-project.tar.gz",
                        background=BackgroundTask(lambda: path.unlink(missing_ok=True)))


@router.post("/api/projects/import", dependencies=[Depends(auth)])
async def import_project(request: Request) -> dict[str, Any]:
    """Import a project export. The request body is the raw .tar.gz bytes
    (streamed to disk — no multipart, no size-of-project RAM spike)."""
    import tempfile
    from pathlib import Path as _P

    from .. import transfer

    fd, tmp = tempfile.mkstemp(suffix=".tar.gz", prefix="hax_import_upload_")
    try:
        with open(fd, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        result = await asyncio.to_thread(transfer.import_project, _P(tmp))
    except transfer.TransferError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ProjectsRootUnset as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    finally:
        _P(tmp).unlink(missing_ok=True)
    return result
