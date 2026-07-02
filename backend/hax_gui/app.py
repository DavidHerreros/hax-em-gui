"""FastAPI application: program catalog, job execution, live logs, file browser.

Run (inside the hax conda env):
    uvicorn hax_gui.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import adapters
from . import files as filebrowser
from . import branches, gpus, projects, registry, settings_store, updates
from .branches import BranchError
from .updater import UpdateBlocked, UpdateBusy, updater
from .command import ValidationError
from .config import settings
from .jobs import JobActive, JobNotFound, JobNotSaved, ProjectNotFound, manager
from .labels import manager as label_manager
from .workflows import WorkflowError, manager as wf_manager
from .projects import ProjectsRootUnset
from .tensorboard import tb_manager
from .viewer import viewer_manager
from .mrc_viewer import mrc_viewer_manager
from .md_viewer import md_viewer_manager

# Hop-by-hop headers must not be forwarded through the TensorBoard proxy.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}

app = FastAPI(title="Hax GUI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    """Force the browser to revalidate static assets so edits show up without a
    hard refresh (StaticFiles still answers 304 when a file is unchanged)."""
    response = await call_next(request)
    if not request.url.path.startswith(("/api", "/ws", "/tb")):
        response.headers["Cache-Control"] = "no-cache"
    return response


# How often the background poller re-checks PyPI / git for newer versions. The
# result is cached on app.state so page loads read it instantly (never block on
# the network); the first check fires shortly after startup.
_UPDATE_POLL_SECONDS = 6 * 60 * 60


@app.on_event("startup")
def _startup() -> None:
    manager.init()
    label_manager.init()
    wf_manager.init()
    app.state.http = httpx.AsyncClient(timeout=None)
    app.state.shutdown = asyncio.Event()
    app.state.updates = {"checked_at": None, "checking": False, "packages": []}
    app.state.update_poller = asyncio.create_task(_poll_updates())


@app.on_event("shutdown")
async def _shutdown() -> None:
    app.state.shutdown.set()  # unblock idle log WebSockets so the server can exit
    poller = getattr(app.state, "update_poller", None)
    if poller is not None:
        poller.cancel()
    tb_manager.shutdown()
    viewer_manager.shutdown()
    mrc_viewer_manager.shutdown()
    md_viewer_manager.shutdown()
    await app.state.http.aclose()


async def _run_update_check() -> None:
    """Refresh the cached update status once (off the event loop)."""
    app.state.updates["checking"] = True
    try:
        packages = await asyncio.to_thread(updates.check_updates)
        app.state.updates = {
            "checked_at": time.time(),
            "checking": False,
            "packages": packages,
        }
    except Exception:  # best-effort: keep the last good result on failure
        app.state.updates["checking"] = False


async def _poll_updates() -> None:
    """Background loop: check on startup, then every _UPDATE_POLL_SECONDS."""
    while not app.state.shutdown.is_set():
        await _run_update_check()
        try:
            await asyncio.wait_for(app.state.shutdown.wait(), timeout=_UPDATE_POLL_SECONDS)
        except asyncio.TimeoutError:
            continue
        break  # shutdown was signalled


# -- auth ------------------------------------------------------------------
def auth(authorization: str | None = Header(default=None)) -> None:
    """Header bearer-token check. No token configured => open (dev only)."""
    if not settings.auth_token:
        return
    if authorization != f"Bearer {settings.auth_token}":
        raise HTTPException(status_code=401, detail="invalid or missing token")


# -- models ----------------------------------------------------------------
class JobCreate(BaseModel):
    program: str
    args: dict[str, Any] = {}
    gpu: str | None = None
    name: str | None = None
    # {field_name: {"job_id": ..., "kind": ...}} — fields fed from a prior job.
    bindings: dict[str, dict[str, Any]] = {}
    # When false, prepare the job but leave it "saved" (don't launch it yet).
    launch: bool = True
    # Wipe the shared SSD scratch folder before launching so it's rebuilt fresh
    # (no effect unless the program uses a scratch folder and a root is configured).
    recreate_ssd: bool = False
    # Reuse an existing job (same id / list position) instead of inserting a new
    # one — set when editing a saved/finished job or re-running a job just saved
    # from the launcher, so Save-then-Run and Edit don't spawn duplicates.
    replace_job_id: str | None = None


# -- catalog ---------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "programs": len(registry.all_programs()), "schema_errors": registry.errors()}


@app.get("/api/programs", dependencies=[Depends(auth)])
def programs() -> list[dict[str, Any]]:
    return registry.list_programs()


@app.get("/api/programs/{name}", dependencies=[Depends(auth)])
def program(name: str) -> dict[str, Any]:
    schema = registry.get_schema(name)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"unknown program '{name}'")
    # The project manager auto-fills the output + SSD scratch dirs, so the form
    # hides those args. `bindable` marks fields that can be fed from a previous
    # job's output, and `producers` lists which programs can feed each kind.
    return {
        **schema,
        "output_arg": registry.output_arg(name),
        "ssd_arg": registry.ssd_arg(name),
        "bindable": adapters.bindable_fields(name),
        "bindable_multi": adapters.multi_bindable_fields(name),
        "producers": adapters.producers_for(name),
        "produces": adapters.produces(name),
    }


@app.get("/api/gpus", dependencies=[Depends(auth)])
def list_gpus() -> list[dict[str, Any]]:
    return gpus.list_gpus()


@app.get("/api/updates", dependencies=[Depends(auth)])
def get_updates() -> dict[str, Any]:
    """Cached update status for hax-em / hax-em-gui (refreshed in the background)."""
    return app.state.updates


@app.post("/api/updates/check", dependencies=[Depends(auth)])
async def refresh_updates() -> dict[str, Any]:
    """Force an immediate re-check (used by the manual 'Check now' affordance)."""
    if not app.state.updates.get("checking"):
        await _run_update_check()
    return app.state.updates


class UpdateRun(BaseModel):
    package: str
    # "full" = pull + install deps + restart/finish; "pull" = pull only, then the
    # GUI server closes so the user installs deps and restarts by hand (devel).
    mode: str = "full"


@app.get("/api/updates/run", dependencies=[Depends(auth)])
def update_run_state() -> dict[str, Any]:
    return updater.state()


@app.post("/api/updates/run", dependencies=[Depends(auth)])
async def run_update(body: UpdateRun) -> dict[str, Any]:
    """Install the update for one package (output streams over /ws/updates/log)."""
    try:
        return await updater.run(body.package, body.mode)
    except UpdateBusy:
        raise HTTPException(status_code=409, detail="an update is already running")
    except UpdateBlocked as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.websocket("/ws/updates/log")
async def update_log_ws(websocket: WebSocket) -> None:
    if settings.auth_token and websocket.query_params.get("token") != settings.auth_token:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    queue, snapshot = updater.subscribe()
    try:
        state = updater.state()
        await websocket.send_json({"type": "snapshot", "data": snapshot, "status": state["status"]})
        if state["status"] != "running":
            # Nothing streaming: replay the final status so a late opener sees it.
            await websocket.send_json({
                "type": "status", "status": state["status"], "return_code": state["return_code"],
                "restarting": state["restarting"], "closing": state["closing"], "note": state["note"],
            })
            return
        shutdown = websocket.app.state.shutdown
        while True:
            getter = asyncio.ensure_future(queue.get())
            stopper = asyncio.ensure_future(shutdown.wait())
            done, pending = await asyncio.wait({getter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if stopper in done:
                break
            message = getter.result()
            await websocket.send_json(message)
            if message.get("type") == "status" and message.get("status") != "running":
                break
    except WebSocketDisconnect:
        pass
    finally:
        updater.unsubscribe(queue)


@app.get("/api/files", dependencies=[Depends(auth)])
def browse(path: str | None = None) -> dict[str, Any]:
    try:
        return filebrowser.list_dir(path)
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# -- settings --------------------------------------------------------------
class SettingsUpdate(BaseModel):
    projects_root: str | None = None
    ssd_scratch_root: str | None = None


@app.get("/api/settings", dependencies=[Depends(auth)])
def get_settings() -> dict[str, Any]:
    return settings_store.load()


@app.put("/api/settings", dependencies=[Depends(auth)])
def put_settings(body: SettingsUpdate) -> dict[str, Any]:
    return settings_store.update(body.model_dump(exclude_none=True))


# -- projects --------------------------------------------------------------
class ProjectCreate(BaseModel):
    name: str


class ProjectRename(BaseModel):
    name: str


def _with_run_counts(projs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each project with the number of runs it holds (for the cards)."""
    for proj in projs:
        proj["runs"] = len(manager.list_for_project(proj["id"]))
    return projs


@app.get("/api/projects", dependencies=[Depends(auth)])
def list_projects() -> list[dict[str, Any]]:
    try:
        return _with_run_counts(projects.list_projects())
    except ProjectsRootUnset:
        return []


@app.post("/api/projects", dependencies=[Depends(auth)])
def create_project(body: ProjectCreate) -> dict[str, Any]:
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="project name required")
    try:
        return projects.create(body.name)
    except ProjectsRootUnset as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/projects/{project_id}", dependencies=[Depends(auth)])
def get_project(project_id: str) -> dict[str, Any]:
    proj = _project_or_404(project_id)
    return proj


@app.put("/api/projects/{project_id}", dependencies=[Depends(auth)])
def rename_project(project_id: str, body: ProjectRename) -> dict[str, Any]:
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="project name required")
    _project_or_404(project_id)
    proj = projects.rename(project_id, body.name)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"unknown project '{project_id}'")
    return proj


@app.delete("/api/projects/{project_id}", dependencies=[Depends(auth)])
def delete_project(project_id: str) -> dict[str, bool]:
    _project_or_404(project_id)
    try:
        manager.delete_for_project(project_id)
    except JobActive:
        raise HTTPException(status_code=409, detail="cancel the project's running jobs before deleting it")
    projects.delete(project_id)
    return {"deleted": True}


@app.get("/api/projects/{project_id}/jobs", dependencies=[Depends(auth)])
def project_jobs(project_id: str) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    return _attach_labels(manager.list_for_project(project_id))


@app.get("/api/projects/{project_id}/sources", dependencies=[Depends(auth)])
def project_sources(project_id: str, kind: str) -> list[dict[str, Any]]:
    """Jobs in this project whose output can feed a `kind` input (for chaining)."""
    _project_or_404(project_id)
    return adapters.compatible_sources(kind, manager.list_for_project(project_id))


@app.post("/api/projects/{project_id}/jobs", dependencies=[Depends(auth)])
async def create_project_job(project_id: str, body: JobCreate) -> dict[str, Any]:
    # Reuse an existing job in place only when it's a real, non-active job of this
    # project; otherwise fall back to creating a fresh one (e.g. it was deleted or
    # is already running), so a stale replace target can never error or clobber a run.
    replace = body.replace_job_id
    if replace:
        existing = manager.get(replace)
        if existing is None or existing.get("project_id") != project_id or existing.get("active"):
            replace = None
    try:
        return await manager.create(body.program, body.args, body.gpu, project_id, body.name, body.bindings, body.launch,
                                    replace_job_id=replace, recreate_ssd=body.recreate_ssd)
    except ProjectNotFound:
        raise HTTPException(status_code=404, detail=f"unknown project '{project_id}'")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})


def _project_or_404(project_id: str) -> dict[str, Any]:
    try:
        proj = projects.get(project_id)
    except ProjectsRootUnset:
        proj = None
    if proj is None:
        raise HTTPException(status_code=404, detail=f"unknown project '{project_id}'")
    return proj


# -- jobs ------------------------------------------------------------------
@app.get("/api/jobs", dependencies=[Depends(auth)])
def list_jobs(limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
    return _with_project_names(manager.list(limit=limit, status=status))


def _with_project_names(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the project display name + labels to each job (best-effort)."""
    names: dict[str, str | None] = {}
    for job in jobs:
        pid = job.get("project_id")
        if pid and pid not in names:
            try:
                proj = projects.get(pid)
            except ProjectsRootUnset:
                proj = None
            names[pid] = proj["name"] if proj else None
        job["project_name"] = names.get(pid) if pid else None
    return _attach_labels(jobs)


def _attach_labels(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach each job's labels (one batched query)."""
    by_job = label_manager.labels_for_many([j["id"] for j in jobs])
    for job in jobs:
        job["labels"] = by_job.get(job["id"], [])
    return jobs


@app.get("/api/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _with_project_names([job])[0]


@app.get("/api/jobs/{job_id}/log", dependencies=[Depends(auth)])
def job_log(job_id: str, max_lines: int | None = None) -> dict[str, Any]:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"lines": manager.tail(job_id, max_lines)}


class JobRename(BaseModel):
    name: str | None = None


@app.put("/api/jobs/{job_id}", dependencies=[Depends(auth)])
def rename_job(job_id: str, body: JobRename) -> dict[str, Any]:
    try:
        return _with_project_names([manager.rename(job_id, body.name or "")])[0]
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")


@app.post("/api/jobs/{job_id}/rerun", dependencies=[Depends(auth)])
async def rerun_job(job_id: str) -> dict[str, Any]:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    try:
        return await manager.rerun(job_id)
    except JobActive:
        raise HTTPException(status_code=409, detail="job is already running")
    except ProjectNotFound:
        raise HTTPException(status_code=409, detail="the job's project no longer exists")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})


@app.post("/api/jobs/{job_id}/start", dependencies=[Depends(auth)])
async def start_job(job_id: str) -> dict[str, Any]:
    try:
        return _with_project_names([await manager.start(job_id)])[0]
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")
    except JobActive:
        raise HTTPException(status_code=409, detail="job is already running")
    except JobNotSaved:
        raise HTTPException(status_code=409, detail="only saved jobs can be started")


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(auth)])
async def cancel_job(job_id: str) -> dict[str, Any]:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    await manager.cancel(job_id)
    return manager.get(job_id)


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(auth)])
def delete_job(job_id: str) -> dict[str, bool]:
    try:
        manager.delete(job_id)
    except JobActive:
        raise HTTPException(status_code=409, detail="cancel the job before deleting it")
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")
    return {"deleted": True}


@app.websocket("/ws/jobs/{job_id}/log")
async def job_log_ws(websocket: WebSocket, job_id: str) -> None:
    # WebSockets can't send custom headers from the browser, so auth via ?token=.
    if settings.auth_token and websocket.query_params.get("token") != settings.auth_token:
        await websocket.close(code=4401)
        return
    job = manager.get(job_id)
    if job is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    # subscribe() registers our queue atomically; we then snapshot from the full
    # persisted log (read_log runs with no await in between, so no streamed chunk is
    # dropped or duplicated). The in-memory buffer is only a small tail and would
    # lose the earliest lines once tqdm progress output has filled it.
    queue, _ = manager.subscribe(job_id)
    try:
        if not job["active"]:
            # Finished job: replay the full log and close — nothing more will stream.
            manager.unsubscribe(job_id, queue)
            await websocket.send_json({"type": "snapshot", "data": manager.read_log(job_id), "status": job["status"]})
            await websocket.send_json({"type": "status", "status": job["status"], "return_code": job["return_code"]})
            return

        await websocket.send_json({"type": "snapshot", "data": manager.read_log(job_id), "status": job["status"]})
        shutdown = websocket.app.state.shutdown
        while True:
            getter = asyncio.ensure_future(queue.get())
            stopper = asyncio.ensure_future(shutdown.wait())
            done, pending = await asyncio.wait({getter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if stopper in done:
                break  # server is shutting down
            message = getter.result()
            await websocket.send_json(message)
            if message.get("type") == "status" and message.get("status") not in ("running",):
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(job_id, queue)


# -- developer / git branches (editable installs only) ---------------------
class BranchSwitch(BaseModel):
    package: str
    branch: str


class PackageRef(BaseModel):
    package: str


class BranchSource(BaseModel):
    package: str
    path: str | None = None  # empty/None clears the configured source clone


@app.get("/api/branches", dependencies=[Depends(auth)])
async def list_branches() -> list[dict[str, Any]]:
    """Branch status for every watched package (a stub where no checkout exists)."""
    return await asyncio.to_thread(branches.list_all)


@app.post("/api/branches/source", dependencies=[Depends(auth)])
async def set_branch_source(body: BranchSource) -> dict[str, Any]:
    """Point a package at a source clone the install itself doesn't reference."""
    try:
        return await asyncio.to_thread(branches.set_source, body.package, body.path)
    except BranchError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/branches/fetch", dependencies=[Depends(auth)])
async def fetch_branches(body: PackageRef) -> dict[str, Any]:
    """Refresh remote-tracking refs for one package, then return fresh status."""
    info = await asyncio.to_thread(branches.fetch, body.package)
    if info is None:
        raise HTTPException(status_code=404, detail=f"'{body.package}' is not a development install")
    return info


@app.post("/api/branches/switch", dependencies=[Depends(auth)])
async def switch_branch(body: BranchSwitch) -> dict[str, Any]:
    """Check out a branch (refused on a dirty tree or an unknown branch)."""
    if manager.count_active() > 0:
        raise HTTPException(status_code=409, detail="a job is still running — wait for it to finish")
    try:
        result = await asyncio.to_thread(branches.switch, body.package, body.branch)
    except BranchError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    # The new branch may sit at a different version vs its upstream — re-probe.
    await _run_update_check()
    return result


@app.post("/api/server/restart", dependencies=[Depends(auth)])
def restart_server() -> dict[str, Any]:
    """Restart the server in place (e.g. after switching the GUI's own branch).

    Reuses the same detached handshake as a GUI self-update; the client then
    health-polls and reloads. False when launched outside ``hax-gui start``."""
    from . import cli

    return {"scheduled": cli.spawn_restart()}


# -- labels ----------------------------------------------------------------
class LabelBody(BaseModel):
    name: str
    color: str | None = None


class LabelUpdate(BaseModel):
    name: str | None = None
    color: str | None = None


class JobLabels(BaseModel):
    labels: list[str] = []


@app.get("/api/labels", dependencies=[Depends(auth)])
def list_labels() -> list[dict[str, Any]]:
    return label_manager.list()


@app.post("/api/labels", dependencies=[Depends(auth)])
def create_label(body: LabelBody) -> dict[str, Any]:
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="label name required")
    return label_manager.create(body.name, body.color)


@app.put("/api/labels/{lid}", dependencies=[Depends(auth)])
def update_label(lid: str, body: LabelUpdate) -> dict[str, Any]:
    try:
        return label_manager.update(lid, name=body.name, color=body.color)
    except KeyError:
        raise HTTPException(status_code=404, detail="label not found")


@app.delete("/api/labels/{lid}", dependencies=[Depends(auth)])
def delete_label(lid: str) -> dict[str, bool]:
    label_manager.delete(lid)
    return {"deleted": True}


@app.put("/api/jobs/{job_id}/labels", dependencies=[Depends(auth)])
def set_job_labels(job_id: str, body: JobLabels) -> dict[str, Any]:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"labels": label_manager.set_job_labels(job_id, body.labels)}


# -- metrics (TensorBoard) -------------------------------------------------
class TensorBoardStart(BaseModel):
    logdir: str


@app.post("/api/tensorboard", dependencies=[Depends(auth)])
async def start_tensorboard(body: TensorBoardStart) -> dict[str, Any]:
    try:
        return await tb_manager.ensure(body.logdir)
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"TensorBoard failed to start: {exc}")


@app.get("/api/tensorboard", dependencies=[Depends(auth)])
def list_tensorboards() -> list[dict[str, Any]]:
    return tb_manager.list()


def _has_tb_events(workdir: Path) -> bool:
    if not workdir.is_dir():
        return False
    # Check the run folder and a couple of levels down (cheap, bounded).
    for pattern in ("events.out.tfevents.*", "*/events.out.tfevents.*", "*/*/events.out.tfevents.*"):
        if next(workdir.glob(pattern), None) is not None:
            return True
    return False


@app.get("/api/metrics/jobs", dependencies=[Depends(auth)])
def metrics_jobs() -> list[dict[str, Any]]:
    """Runs whose output folder contains TensorBoard event files."""
    out = []
    for job in manager.list(limit=500):
        wd = job.get("workdir")
        if wd and _has_tb_events(Path(wd)):
            out.append({k: job[k] for k in ("id", "program", "workdir", "created_at", "status", "project_id")})
    return _with_project_names(out)


@app.delete("/api/tensorboard/{tb_id}", dependencies=[Depends(auth)])
def stop_tensorboard(tb_id: str) -> dict[str, bool]:
    return {"stopped": tb_manager.stop(tb_id)}


# Reverse proxy to per-instance TensorBoard servers. The instance id is an
# unguessable capability token, so these routes are intentionally not behind the
# bearer auth (the browser can't send headers for iframe sub-resources).
@app.api_route("/tb/{tb_id}", methods=["GET"])
def tb_redirect(tb_id: str) -> RedirectResponse:
    return RedirectResponse(url=f"/tb/{tb_id}/")


@app.api_route("/tb/{tb_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def tb_proxy(tb_id: str, path: str, request: Request) -> Response:
    inst = tb_manager.get(tb_id)
    if inst is None:
        raise HTTPException(status_code=404, detail="unknown TensorBoard session")

    query = request.url.query
    url = f"http://127.0.0.1:{inst['port']}/tb/{tb_id}/{path}" + (f"?{query}" if query else "")
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP and k.lower() != "host"}
    body = await request.body()

    upstream = await request.app.state.http.request(request.method, url, headers=fwd_headers, content=body)
    # httpx already decoded content-encoding, so drop length/encoding and let
    # the Response recompute them.
    out_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() not in ("content-encoding", "content-length")
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=upstream.headers.get("content-type"),
    )


# -- WebGL viewer (per-session subprocess, reverse-proxied) ----------------
class ViewerStart(BaseModel):
    latents: str
    reload: str
    title: str | None = None
    reduced: str | None = None


@app.post("/api/viewer", dependencies=[Depends(auth)])
async def start_viewer(body: ViewerStart) -> dict[str, Any]:
    return await _ensure_viewer(body.latents, body.reload, body.title or "Conformational landscape",
                                reduced=body.reduced)


@app.get("/api/viewer/flexconsensus/{job_id}", dependencies=[Depends(auth)])
def flexconsensus_spaces(job_id: str) -> dict[str, Any]:
    """The consensus output spaces of a FlexConsensus run, each with the decoder
    network auto-detected from the predict run that produced its input space."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("program") != "flexconsensus":
        raise HTTPException(status_code=422, detail="not a FlexConsensus run")
    spaces = _flexconsensus_spaces(job)
    return {"spaces": [{k: v for k, v in s.items() if k != "reload"} for s in spaces]}


@app.post("/api/viewer/from-job/{job_id}", dependencies=[Depends(auth)])
async def start_viewer_from_job(job_id: str, space: int | None = None) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    program = job.get("program", "")
    args = job.get("args", {})
    # FlexConsensus: pick one consensus output space; its input space supplies the
    # full latents to decode, and the decoder network is auto-detected.
    if program == "flexconsensus":
        spaces = _flexconsensus_spaces(job)
        if space is None:
            raise HTTPException(status_code=422,
                                detail={"choose_flexconsensus_space": True,
                                        "spaces": [{k: v for k, v in s.items() if k != "reload"} for s in spaces]})
        sp = next((s for s in spaces if s["index"] == space), None)
        if sp is None:
            raise HTTPException(status_code=422, detail="unknown consensus space")
        if not sp["viewable"]:
            raise HTTPException(status_code=422, detail=sp["reason"] or "this consensus space is not viewable")
        title = f"{job.get('name') or 'FlexConsensus'} · {sp['name']}"
        project_dir = projects.dir_for(job.get("project_id"))
        metadata = _find_latent_metadata(project_dir, sp["input_space"])
        return await _ensure_viewer(sp["input_space"], sp["reload"], title, metadata,
                                    str(project_dir) if project_dir else None, sp["consensus"])
    reload_raw = args.get("pickled_nn") or args.get("reload")
    # Full latents: a direct input (annotate_space), or derived from this run's
    # own output (a predict run's predicted latents).
    latents = args.get("z_space") or args.get("latents") or args.get("latents_file")
    if not latents:
        try:
            latents = adapters.produce("latents", job, settings.data_dir / "viewer_cache" / job_id, "z_space")
        except adapters.AdapterError:
            latents = None
    if not latents or not reload_raw:
        raise HTTPException(
            status_code=422,
            detail="this run has no latents + network to view (need latents/z_space — or a predict "
                   "run's output — plus pickled_nn/reload)",
        )
    reload = str(adapters.checkpoint_dir(program, reload_raw))
    # Optional reduced display space (e.g. a FlexConsensus consensus bound here).
    reduced = args.get("z_space_reduced")
    title = job.get("name") or job.get("program") or "Conformational landscape"
    project_dir = projects.dir_for(job.get("project_id"))
    metadata = _find_latent_metadata(project_dir, latents)
    return await _ensure_viewer(latents, reload, title, metadata,
                                str(project_dir) if project_dir else None, reduced)


def _find_latent_metadata(project_dir: "Path | None", latents: str) -> str | None:
    """Find the project's particle metadata that is row-aligned with these latents.

    The latents `.npy` was materialised from a predict run's `_latent_space` column;
    we recover that metadata (poses/CTF/latent) so layers can be exported with full
    per-particle info. Matched by `_latent_space` presence + row count + first-row value.
    """
    if project_dir is None:
        return None
    try:
        import numpy as np
        from xmipp_metadata.metadata import XmippMetaData

        lp = Path(latents)
        lat = np.loadtxt(lp) if lp.suffix == ".txt" else np.load(lp)
        lat = np.atleast_2d(np.asarray(lat, dtype=np.float32))
        for p in sorted(project_dir.rglob("*")):
            if p.suffix.lower() not in (".xmd", ".star"):
                continue
            try:
                md = XmippMetaData(str(p))
                if not md.isMetaDataLabel("latent_space") or len(md) != lat.shape[0]:
                    continue
                first = np.fromstring(str(md.getMetaDataColumns("latent_space")[0]), sep=",")
                if first.size == lat.shape[1] and np.allclose(first, lat[0], atol=1e-3):
                    return str(p)
            except Exception:  # noqa: BLE001 - skip unreadable candidates
                continue
    except Exception:  # noqa: BLE001
        return None
    return None


# Programs whose predict run yields latents + a decoder — derived from the
# adapters maps (single source of truth), so new such programs need no edit here.
def _predict_programs() -> set[str]:
    return set(adapters.latents_decoder_programs())


def _load_latents_array(path: str):
    import numpy as np
    p = Path(path)
    arr = np.loadtxt(p) if p.suffix == ".txt" else np.load(p)
    return np.atleast_2d(np.asarray(arr, dtype=np.float32))


def _predict_latents_array(job: dict[str, Any]):
    """Load a predict run's latents (extracting the metadata column if needed)."""
    src = adapters.source_file_for("latents", job)
    if src is None:
        return None
    try:
        if src.suffix == ".npy":
            return _load_latents_array(str(src))
        out_dir = settings.data_dir / "viewer_cache" / "_cmp"
        produced = adapters.produce("latents", job, out_dir, job["id"])
        return _load_latents_array(produced)
    except Exception:  # noqa: BLE001
        return None


def _find_decoder(input_path: str, predict_jobs: list[dict[str, Any]]):
    """Find the predict job (and its decoder checkpoint) that produced `input_path`,
    by path containment first, then by matching latents (shape + first row)."""
    try:
        ip = Path(input_path).expanduser().resolve()
    except Exception:  # noqa: BLE001
        ip = None
    for j in predict_jobs:
        wd = j.get("workdir")
        net = j["args"].get("reload") or j["args"].get("pickled_nn")
        if wd and net and ip and str(ip).startswith(str(Path(wd).expanduser().resolve())):
            return j, str(adapters.checkpoint_dir(j["program"], net))
    import numpy as np
    try:
        lat = _load_latents_array(input_path)
    except Exception:  # noqa: BLE001
        return None, None
    for j in predict_jobs:
        net = j["args"].get("reload") or j["args"].get("pickled_nn")
        if not net:
            continue
        cand = _predict_latents_array(j)
        if cand is not None and cand.shape == lat.shape and np.allclose(cand[0], lat[0], atol=1e-3):
            return j, str(adapters.checkpoint_dir(j["program"], net))
    return None, None


def _flexconsensus_spaces(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve a FlexConsensus run's consensus output spaces. Each space pairs a
    consensus file (display) with its input latent space (decoded) and the
    auto-detected decoder network from the predict run that produced that input."""
    args = job.get("args", {})
    raw = args.get("input_space")
    entries = raw if isinstance(raw, list) else (str(raw).split() if raw else [])
    workdir = Path(job.get("workdir") or ".")
    pid = job.get("project_id")
    predict = [j for j in (manager.list_for_project(pid) if pid else []) if j["program"] in _predict_programs()]
    spaces = []
    for i, entry in enumerate(entries):
        entry = str(entry)
        # FlexConsensus accepts "NAME:path"; plain paths are auto-named "Input_i".
        # (Linux paths have no ':' so a ':' marks an explicit name.)
        if ":" in entry:
            name, ipath = entry.split(":", 1)
        else:
            name, ipath = f"Input_{i}", entry
        cons = workdir / f"{name}_consensus.npy"
        djob, reload = _find_decoder(ipath, predict)
        ok = bool(djob and reload and cons.exists())
        spaces.append({
            "index": i, "name": name, "input_space": ipath, "consensus": str(cons),
            "consensus_exists": cons.exists(),
            "decoder_job": djob["id"] if djob else None,
            "decoder_program": djob["program"] if djob else None,
            "reload": reload, "viewable": ok,
            "reason": None if ok else ("no consensus output found for this space" if not cons.exists()
                                       else "couldn't auto-detect the decoder network for this space"),
        })
    return spaces


async def _ensure_viewer(latents: str, reload: str, title: str,
                         metadata: str | None = None, project_dir: str | None = None,
                         reduced: str | None = None) -> dict[str, Any]:
    try:
        return await viewer_manager.ensure(latents, reload, title, metadata, project_dir, reduced)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"viewer failed to start: {exc}")


@app.get("/api/viewer", dependencies=[Depends(auth)])
def list_viewers() -> list[dict[str, Any]]:
    return viewer_manager.list()


@app.delete("/api/viewer/{vid}", dependencies=[Depends(auth)])
def stop_viewer(vid: str) -> dict[str, bool]:
    return {"stopped": viewer_manager.stop(vid)}


@app.api_route("/viewer/{vid}", methods=["GET"])
def viewer_redirect(vid: str) -> RedirectResponse:
    return RedirectResponse(url=f"/viewer/{vid}/")


@app.api_route("/viewer/{vid}/{path:path}", methods=["GET", "POST", "OPTIONS", "HEAD"])
async def viewer_proxy(vid: str, path: str, request: Request) -> Response:
    inst = viewer_manager.get(vid)
    if inst is None:
        raise HTTPException(status_code=404, detail="unknown viewer session")
    # The viewer subprocess serves at root; strip the /viewer/<id> prefix.
    query = request.url.query
    url = f"http://127.0.0.1:{inst['port']}/{path}" + (f"?{query}" if query else "")
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP and k.lower() != "host"}
    body = await request.body()
    upstream = await request.app.state.http.request(request.method, url, headers=fwd_headers, content=body)
    out_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() not in ("content-encoding", "content-length")
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=upstream.headers.get("content-type"),
    )


# -- Map (.mrc) viewer: inspect any volume a job produced -------------------
_MAP_SUFFIXES = (".mrc", ".map", ".vol")
_MD_SUFFIXES = (".xmd", ".star", ".cs")


def _list_job_files(job_id: str, suffixes: tuple[str, ...]) -> dict[str, Any]:
    """List a job's run-folder files whose suffix is in `suffixes` (recursive)."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    wd = job.get("workdir")
    if not wd:
        return {"workdir": None, "files": []}
    root = Path(wd)
    out: list[dict[str, Any]] = []
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in suffixes:
                try:
                    st = p.stat()
                except OSError:
                    continue
                out.append({
                    "path": str(p),
                    "name": str(p.relative_to(root)),
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                })
    return {"workdir": str(root), "files": out}


def _job_arg_paths(job: dict[str, Any]) -> set[str]:
    """Resolved file paths referenced in a job's argument values. Handles plain
    strings, lists, whitespace-separated multi-values and the ``NAME:path`` form
    (e.g. FlexConsensus input_space). Used to authorise viewing a run's *inputs*."""
    out: set[str] = set()
    args = job.get("args") or {}
    for v in args.values():
        items = v if isinstance(v, list) else [v]
        for it in items:
            if not isinstance(it, (str, int, float)):
                continue
            for tok in str(it).split():
                cand = tok
                if ":" in tok:
                    after = tok.split(":", 1)[1]
                    if after and Path(after).expanduser().exists():
                        cand = after
                try:
                    p = Path(cand).expanduser()
                    if p.is_file():
                        out.add(str(p.resolve()))
                except (OSError, RuntimeError):
                    continue
    return out


def _list_job_input_files(job_id: str, suffixes: tuple[str, ...]) -> dict[str, Any]:
    """List a job's *input* files (from its args) whose suffix is in `suffixes`."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    out: list[dict[str, Any]] = []
    for path in sorted(_job_arg_paths(job)):
        p = Path(path)
        if p.suffix.lower() not in suffixes:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"path": path, "name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    return {"files": out}


def _job_file(job_id: str, path: str, suffixes: tuple[str, ...]) -> tuple[dict[str, Any], Path]:
    """Validate that `path` is a file of the given kind that this job legitimately
    references — either an output inside its run folder or an input in its args
    (so no arbitrary filesystem access is granted through the from-job routes)."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="invalid path")
    wd = job.get("workdir")
    in_workdir = False
    if wd:
        root = Path(wd).resolve()
        in_workdir = root == target or root in target.parents
    if not in_workdir and str(target) not in _job_arg_paths(job):
        raise HTTPException(status_code=403, detail="path is not one of this job's inputs or outputs")
    if not target.is_file() or target.suffix.lower() not in suffixes:
        raise HTTPException(status_code=404, detail="file not found")
    return job, target


@app.get("/api/jobs/{job_id}/mrcs", dependencies=[Depends(auth)])
def job_mrcs(job_id: str) -> dict[str, Any]:
    """List the volume files (.mrc/.map/.vol) inside a job's run folder, so any map
    it produced can be inspected (works while running and after it finishes)."""
    return _list_job_files(job_id, _MAP_SUFFIXES)


@app.get("/api/jobs/{job_id}/inputs", dependencies=[Depends(auth)])
def job_inputs(job_id: str) -> dict[str, Any]:
    """Volume + metadata files a job took as *inputs* (referenced in its args), so
    they can be inspected in the same viewers as the outputs."""
    return {
        "maps": _list_job_input_files(job_id, _MAP_SUFFIXES)["files"],
        "metadata": _list_job_input_files(job_id, _MD_SUFFIXES)["files"],
    }


async def _ensure_mrc_viewer(mrc: str, title: str) -> dict[str, Any]:
    try:
        return await mrc_viewer_manager.ensure(mrc, title)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"map viewer failed to start: {exc}")


class MrcViewerStart(BaseModel):
    path: str
    title: str | None = None


@app.post("/api/mrcviewer", dependencies=[Depends(auth)])
async def start_mrc_viewer(body: MrcViewerStart) -> dict[str, Any]:
    """Open the map viewer for any .mrc within the sandboxed file roots."""
    try:
        path = filebrowser.resolve(body.path)   # confines to HAX_GUI_FILE_ROOTS
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return await _ensure_mrc_viewer(str(path), body.title or path.name)


@app.post("/api/mrcviewer/from-job/{job_id}", dependencies=[Depends(auth)])
async def start_mrc_viewer_from_job(job_id: str, path: str) -> dict[str, Any]:
    """Open the map viewer for a .mrc this job produced or used (the path must be
    one of the job's inputs or outputs — no arbitrary filesystem access here)."""
    job, target = _job_file(job_id, path, _MAP_SUFFIXES)
    title = f"{job.get('name') or job.get('program') or 'map'} · {target.name}"
    return await _ensure_mrc_viewer(str(target), title)


@app.get("/api/mrcviewer", dependencies=[Depends(auth)])
def list_mrc_viewers() -> list[dict[str, Any]]:
    return mrc_viewer_manager.list()


@app.delete("/api/mrcviewer/{vid}", dependencies=[Depends(auth)])
def stop_mrc_viewer(vid: str) -> dict[str, bool]:
    return {"stopped": mrc_viewer_manager.stop(vid)}


@app.api_route("/mrcviewer/{vid}", methods=["GET"])
def mrc_viewer_redirect(vid: str) -> RedirectResponse:
    return RedirectResponse(url=f"/mrcviewer/{vid}/")


@app.api_route("/mrcviewer/{vid}/{path:path}", methods=["GET", "POST", "OPTIONS", "HEAD"])
async def mrc_viewer_proxy(vid: str, path: str, request: Request) -> Response:
    inst = mrc_viewer_manager.get(vid)
    if inst is None:
        raise HTTPException(status_code=404, detail="unknown map-viewer session")
    query = request.url.query
    url = f"http://127.0.0.1:{inst['port']}/{path}" + (f"?{query}" if query else "")
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP and k.lower() != "host"}
    body = await request.body()
    upstream = await request.app.state.http.request(request.method, url, headers=fwd_headers, content=body)
    out_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() not in ("content-encoding", "content-length")
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=upstream.headers.get("content-type"),
    )


# -- Metadata (.xmd/.star/.cs) viewer: browse any table a job produced ------
async def _ensure_md_viewer(md: str, title: str) -> dict[str, Any]:
    try:
        return await md_viewer_manager.ensure(md, title)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"metadata viewer failed to start: {exc}")


class MdViewerStart(BaseModel):
    path: str
    title: str | None = None


@app.get("/api/jobs/{job_id}/metadata", dependencies=[Depends(auth)])
def job_metadata(job_id: str) -> dict[str, Any]:
    """List the metadata files (.xmd/.star/.cs) inside a job's run folder."""
    return _list_job_files(job_id, _MD_SUFFIXES)


@app.post("/api/mdviewer", dependencies=[Depends(auth)])
async def start_md_viewer(body: MdViewerStart) -> dict[str, Any]:
    """Open the metadata viewer for any .xmd/.star/.cs within the file roots."""
    try:
        path = filebrowser.resolve(body.path)   # confines to HAX_GUI_FILE_ROOTS
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return await _ensure_md_viewer(str(path), body.title or path.name)


@app.post("/api/mdviewer/from-job/{job_id}", dependencies=[Depends(auth)])
async def start_md_viewer_from_job(job_id: str, path: str) -> dict[str, Any]:
    """Open the metadata viewer for a table this job produced or used (the path must
    be one of the job's inputs or outputs)."""
    job, target = _job_file(job_id, path, _MD_SUFFIXES)
    title = f"{job.get('name') or job.get('program') or 'metadata'} · {target.name}"
    return await _ensure_md_viewer(str(target), title)


@app.get("/api/mdviewer", dependencies=[Depends(auth)])
def list_md_viewers() -> list[dict[str, Any]]:
    return md_viewer_manager.list()


@app.delete("/api/mdviewer/{vid}", dependencies=[Depends(auth)])
def stop_md_viewer(vid: str) -> dict[str, bool]:
    return {"stopped": md_viewer_manager.stop(vid)}


@app.api_route("/mdviewer/{vid}", methods=["GET"])
def md_viewer_redirect(vid: str) -> RedirectResponse:
    return RedirectResponse(url=f"/mdviewer/{vid}/")


@app.api_route("/mdviewer/{vid}/{path:path}", methods=["GET", "POST", "OPTIONS", "HEAD"])
async def md_viewer_proxy(vid: str, path: str, request: Request) -> Response:
    inst = md_viewer_manager.get(vid)
    if inst is None:
        raise HTTPException(status_code=404, detail="unknown metadata-viewer session")
    query = request.url.query
    url = f"http://127.0.0.1:{inst['port']}/{path}" + (f"?{query}" if query else "")
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP and k.lower() != "host"}
    body = await request.body()
    upstream = await request.app.state.http.request(request.method, url, headers=fwd_headers, content=body)
    out_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() not in ("content-encoding", "content-length")
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=upstream.headers.get("content-type"),
    )


# -- workflows (visual pipeline builder + DAG engine) ----------------------
@app.get("/api/workflows/catalog", dependencies=[Depends(auth)])
def workflow_catalog() -> list[dict[str, Any]]:
    """Program palette for the canvas: each entry's bindable input fields (input
    ports) and the artifact kinds it produces (output ports)."""
    out = []
    for p in registry.list_programs():
        out.append({
            **p,
            "bindable": adapters.bindable_fields(p["name"]),
            "bindable_multi": adapters.multi_bindable_fields(p["name"]),
            "produces": adapters.produces(p["name"]),
        })
    return out


class WorkflowBody(BaseModel):
    project_id: str | None = None
    name: str | None = None
    graph: dict[str, Any] = {"nodes": [], "edges": []}


class WorkflowUpdate(BaseModel):
    name: str | None = None
    graph: dict[str, Any] | None = None


class WorkflowRun(BaseModel):
    # Optional subset of node ids to run (a single node or a selection); when
    # omitted the whole workflow runs.
    nodes: list[str] | None = None


class WorkflowFromProject(BaseModel):
    name: str | None = None


@app.get("/api/workflows", dependencies=[Depends(auth)])
def list_workflows(project_id: str | None = None) -> list[dict[str, Any]]:
    return wf_manager.list(project_id)


@app.post("/api/workflows", dependencies=[Depends(auth)])
def create_workflow(body: WorkflowBody) -> dict[str, Any]:
    return wf_manager.save(body.project_id, body.name, body.graph)


@app.get("/api/workflows/{wid}", dependencies=[Depends(auth)])
def get_workflow(wid: str) -> dict[str, Any]:
    # Reconcile a project workflow with its jobs' live statuses on load (no-op for
    # standalone workflows / while running), so stale node states self-heal.
    wf = wf_manager.reconcile(wid)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    return wf


@app.put("/api/workflows/{wid}", dependencies=[Depends(auth)])
def update_workflow(wid: str, body: WorkflowUpdate) -> dict[str, Any]:
    try:
        return wf_manager.update(wid, name=body.name, graph=body.graph)
    except KeyError:
        raise HTTPException(status_code=404, detail="workflow not found")


@app.delete("/api/workflows/{wid}", dependencies=[Depends(auth)])
def delete_workflow(wid: str) -> dict[str, str]:
    wf_manager.delete(wid)
    return {"status": "deleted"}


@app.post("/api/workflows/{wid}/validate", dependencies=[Depends(auth)])
def validate_workflow(wid: str) -> dict[str, Any]:
    wf = wf_manager.get(wid)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    return {"errors": wf_manager.validate(wf["graph"])}


@app.post("/api/workflows/{wid}/run", dependencies=[Depends(auth)])
async def run_workflow(wid: str, body: WorkflowRun = WorkflowRun()) -> dict[str, Any]:
    try:
        return await wf_manager.run(wid, body.nodes)
    except KeyError:
        raise HTTPException(status_code=404, detail="workflow not found")
    except WorkflowError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})


@app.post("/api/workflows/from-project/{project_id}", dependencies=[Depends(auth)])
def workflow_from_project(project_id: str, body: WorkflowFromProject = WorkflowFromProject()) -> dict[str, Any]:
    """Open a project as a *live* workflow (get-or-create): nodes = the project's
    jobs, edges inferred from input bindings, kept in sync with the project. Editing
    / running / labeling this workflow modifies the actual project."""
    proj = _project_or_404(project_id)
    return wf_manager.open_project(project_id, body.name or proj["name"])


@app.post("/api/workflows/{wid}/save-as-project", dependencies=[Depends(auth)])
def workflow_save_as_project(wid: str, body: WorkflowFromProject = WorkflowFromProject()) -> dict[str, Any]:
    """Create a new project and bind this workflow to it as the project's live
    view, so further edits/runs of the workflow populate the new project."""
    wf = wf_manager.get(wid)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    name = (body.name or wf.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="project name required")
    try:
        return wf_manager.save_as_project(wid, name)
    except ProjectsRootUnset as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# -- static SPA (mounted last so /api and /ws take precedence) -------------
# The frontend is plain ES modules + CSS (no build step), so we serve the
# directory directly. Same-origin => no CORS/proxy needed for production use.
def mount_spa() -> None:
    frontend = settings.frontend_dir
    if (frontend / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="spa")


mount_spa()
