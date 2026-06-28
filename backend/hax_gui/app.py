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
from .labels import manager as label_manager
from .workflows import WorkflowError, manager as wf_manager
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
    label_manager.init()
    wf_manager.init()
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
        "bindable_multi": adapters.multi_bindable_fields(name),
        "producers": adapters.producers_for(name),
        "produces": adapters.produces(name),
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
    return _attach_labels(manager.list_for_project(project_id))


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


_PREDICT_PROGRAMS = {"hetsiren", "zernike3deep", "reconsiren"}


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
    predict = [j for j in (manager.list_for_project(pid) if pid else []) if j["program"] in _PREDICT_PROGRAMS]
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
    wf = wf_manager.get(wid)
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
