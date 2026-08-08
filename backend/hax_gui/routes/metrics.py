"""Metrics routes: TensorBoard instances + proxy, and the native event-file reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from .. import metrics as metrics_reader
from ..jobs import manager
from .. import files as filebrowser
from ..tensorboard import tb_manager
from .shared import auth, has_tb_events as _has_tb_events, proxy as _proxy, with_project_names as _with_project_names

router = APIRouter()

# -- metrics (TensorBoard) -------------------------------------------------
class TensorBoardStart(BaseModel):
    logdir: str


@router.post("/api/tensorboard", dependencies=[Depends(auth)])
async def start_tensorboard(body: TensorBoardStart) -> dict[str, Any]:
    try:
        return await tb_manager.ensure(body.logdir)
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"TensorBoard failed to start: {exc}")


@router.get("/api/tensorboard", dependencies=[Depends(auth)])
def list_tensorboards() -> list[dict[str, Any]]:
    return tb_manager.list()


@router.get("/api/metrics/jobs", dependencies=[Depends(auth)])
def metrics_jobs() -> list[dict[str, Any]]:
    """Runs whose output folder contains TensorBoard event files."""
    out = []
    for job in manager.list(limit=500):
        wd = job.get("workdir")
        if wd and _has_tb_events(Path(wd)):
            # started_at/finished_at/active let the picker date the run itself rather than
            # the row: created_at survives a rerun, so alone it dates a run that is gone.
            out.append({k: job[k] for k in ("id", "program", "name", "workdir", "created_at",
                                            "started_at", "finished_at", "active", "status", "project_id")})
    return _with_project_names(out)


@router.delete("/api/tensorboard/{tb_id}", dependencies=[Depends(auth)])
def stop_tensorboard(tb_id: str) -> dict[str, bool]:
    return {"stopped": tb_manager.stop(tb_id)}


# -- metrics (native reader over the same event files) ---------------------
# The GUI draws the curves/images/latent space itself; TensorBoard stays available
# via the endpoints above, opened in its own browser tab. Reading is blocking I/O
# (protobuf decode), so every call hops to the threadpool.
def _metrics_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except metrics_reader.TensorBoardMissing as exc:
        raise HTTPException(status_code=501, detail=str(exc))


@router.get("/api/metrics/manifest", dependencies=[Depends(auth)])
async def metrics_manifest(logdir: str) -> dict[str, Any]:
    """Which tags a run logged, of what kind, in which sub-runs, at which steps."""
    return await run_in_threadpool(_metrics_call, metrics_reader.manifest, logdir)


@router.get("/api/metrics/scalars", dependencies=[Depends(auth)])
async def metrics_scalars(logdir: str, tags: str | None = None) -> dict[str, Any]:
    want = [t for t in tags.split("\n") if t] if tags else None
    return await run_in_threadpool(_metrics_call, metrics_reader.scalars, logdir, want)


@router.get("/api/metrics/image", dependencies=[Depends(auth)])
async def metrics_image(logdir: str, run: str, tag: str, frame: int | None = None) -> Response:
    png = await run_in_threadpool(_metrics_call, metrics_reader.image, logdir, run, tag, frame)
    return Response(content=png, media_type="image/png")


@router.get("/api/metrics/embedding", dependencies=[Depends(auth)])
async def metrics_embedding(logdir: str, step: int, tag: str | None = None) -> dict[str, Any]:
    """One epoch of the projector's latent space. Re-read from disk every call, so
    epochs written after the page loaded just show up."""
    return await run_in_threadpool(_metrics_call, metrics_reader.embedding, logdir, step, tag)


@router.get("/api/metrics/sprite", dependencies=[Depends(auth)])
async def metrics_sprite(logdir: str, step: int, tag: str | None = None) -> Response:
    png = await run_in_threadpool(_metrics_call, metrics_reader.sprite, logdir, step, tag)
    return Response(content=png, media_type="image/png")


@router.delete("/api/metrics/cache", dependencies=[Depends(auth)])
async def metrics_drop(logdir: str) -> dict[str, bool]:
    """Release a log dir's accumulators (its image reservoir can be sizeable)."""
    return {"dropped": await run_in_threadpool(_metrics_call, metrics_reader.drop, logdir)}


@router.post("/api/tensorboard/{tb_id}/restart", dependencies=[Depends(auth)])
async def restart_tensorboard(tb_id: str) -> dict[str, Any]:
    """Respawn TB on the same logs — the only way to pick up new projector epochs."""
    try:
        inst = await tb_manager.restart(tb_id)
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"TensorBoard failed to start: {exc}")
    if inst is None:
        raise HTTPException(status_code=404, detail="tensorboard not found")
    return inst


# Reverse proxy to per-instance TensorBoard servers. The instance id is an
# unguessable capability token, so these routes are intentionally not behind the
# bearer auth (the browser can't send headers for iframe sub-resources).
@router.api_route("/tb/{tb_id}", methods=["GET"])
def tb_redirect(tb_id: str) -> RedirectResponse:
    return RedirectResponse(url=f"/tb/{tb_id}/")


@router.api_route("/tb/{tb_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def tb_proxy(tb_id: str, path: str, request: Request) -> Response:
    inst = tb_manager.get(tb_id)
    if inst is None:
        raise HTTPException(status_code=404, detail="unknown TensorBoard session")

    query = request.url.query
    url = f"http://127.0.0.1:{inst['port']}/tb/{tb_id}/{path}" + (f"?{query}" if query else "")
    return await _proxy(request, url)

