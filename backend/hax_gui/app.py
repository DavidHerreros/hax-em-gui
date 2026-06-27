"""FastAPI application: program catalog, job execution, live logs, file browser.

Run (inside the hax conda env):
    uvicorn hax_gui.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
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
from . import gpus, projects, registry, settings_store
from .command import ValidationError
from .config import settings
from .jobs import JobActive, JobNotFound, JobNotSaved, ProjectNotFound, manager
from .projects import ProjectsRootUnset
from .tensorboard import tb_manager
from .viewer import viewer_manager

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


@app.on_event("startup")
def _startup() -> None:
    manager.init()
    app.state.http = httpx.AsyncClient(timeout=None)
    app.state.shutdown = asyncio.Event()


@app.on_event("shutdown")
async def _shutdown() -> None:
    app.state.shutdown.set()  # unblock idle log WebSockets so the server can exit
    tb_manager.shutdown()
    viewer_manager.shutdown()
    await app.state.http.aclose()


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
        "producers": adapters.producers_for(name),
    }


@app.get("/api/gpus", dependencies=[Depends(auth)])
def list_gpus() -> list[dict[str, Any]]:
    return gpus.list_gpus()


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
    return manager.list_for_project(project_id)


@app.get("/api/projects/{project_id}/sources", dependencies=[Depends(auth)])
def project_sources(project_id: str, kind: str) -> list[dict[str, Any]]:
    """Jobs in this project whose output can feed a `kind` input (for chaining)."""
    _project_or_404(project_id)
    return adapters.compatible_sources(kind, manager.list_for_project(project_id))


@app.post("/api/projects/{project_id}/jobs", dependencies=[Depends(auth)])
async def create_project_job(project_id: str, body: JobCreate) -> dict[str, Any]:
    try:
        return await manager.create(body.program, body.args, body.gpu, project_id, body.name, body.bindings, body.launch)
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
    """Attach the project display name to each job (best-effort)."""
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
    queue, snapshot = manager.subscribe(job_id)
    try:
        if not job["active"]:
            # Finished job: replay the full persisted log (the in-memory buffer is
            # only a tail) and close — nothing more will stream.
            manager.unsubscribe(job_id, queue)
            await websocket.send_json({"type": "snapshot", "data": manager.read_log(job_id), "status": job["status"]})
            await websocket.send_json({"type": "status", "status": job["status"], "return_code": job["return_code"]})
            return

        await websocket.send_json({"type": "snapshot", "data": snapshot, "status": job["status"]})
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


@app.post("/api/viewer", dependencies=[Depends(auth)])
async def start_viewer(body: ViewerStart) -> dict[str, Any]:
    return await _ensure_viewer(body.latents, body.reload, body.title or "Conformational landscape")


@app.post("/api/viewer/from-job/{job_id}", dependencies=[Depends(auth)])
async def start_viewer_from_job(job_id: str) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    args = job.get("args", {})
    latents = args.get("z_space") or args.get("latents") or args.get("latents_file")
    reload = args.get("pickled_nn") or args.get("reload")
    if not latents or not reload:
        raise HTTPException(
            status_code=422,
            detail="this run has no latents + network to view (need z_space/latents and pickled_nn/reload)",
        )
    title = job.get("name") or job.get("program") or "Conformational landscape"
    project_dir = projects.dir_for(job.get("project_id"))
    metadata = _find_latent_metadata(project_dir, latents)
    return await _ensure_viewer(latents, reload, title, metadata, str(project_dir) if project_dir else None)


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


async def _ensure_viewer(latents: str, reload: str, title: str,
                         metadata: str | None = None, project_dir: str | None = None) -> dict[str, Any]:
    try:
        return await viewer_manager.ensure(latents, reload, title, metadata, project_dir)
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


# -- static SPA (mounted last so /api and /ws take precedence) -------------
# The frontend is plain ES modules + CSS (no build step), so we serve the
# directory directly. Same-origin => no CORS/proxy needed for production use.
def mount_spa() -> None:
    frontend = settings.frontend_dir
    if (frontend / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="spa")


mount_spa()
