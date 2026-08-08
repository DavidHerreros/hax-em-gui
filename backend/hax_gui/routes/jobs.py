"""Job routes: list/detail/log, rename, rerun/continue/start/cancel/delete,
dependents, copy-prefill, and the live log WebSocket."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .. import adapters, cascade
from ..command import ValidationError
from ..jobs import ACTIVE, JobActive, JobNotFound, JobNotResumable, JobNotSaved, ProjectNotFound, manager
from ..labels import manager as label_manager
from ..workflows import manager as wf_manager
from .shared import (
    RerunBody, apply_cascade as _apply_cascade, auth, has_tb_events as _has_tb_events,
    queue_cascade as _queue_cascade, with_project_names as _with_project_names,
    ws_authorized as _ws_authorized,
)

router = APIRouter()

# -- jobs ------------------------------------------------------------------
@router.get("/api/jobs", dependencies=[Depends(auth)])
def list_jobs(limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
    return _with_project_names(manager.list(limit=limit, status=status))


@router.get("/api/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # Whether this run left an intermediate checkpoint its program will auto-resume from
    # (drives the "Continue" buttons). Only meaningful once the run has finished, so
    # skip the disk check for active/saved jobs.
    # A stale run is excluded too: continuing reuses its stored command, whose inputs
    # point into a source run folder that has since been rebuilt (see cascade.py).
    job["resumable"] = False if job.get("active") or job["status"] == "saved" or job.get("stale") else \
        adapters.is_resumable(job["program"], job.get("workdir"))
    # Did this run write any event files? Drives the "Metrics" button. A running job
    # gets one too — the native metrics view follows a live run.
    wd = job.get("workdir")
    job["has_metrics"] = bool(wd) and job["status"] != "saved" and _has_tb_events(Path(wd))
    return _with_project_names([job])[0]


@router.get("/api/jobs/{job_id}/log", dependencies=[Depends(auth)])
def job_log(job_id: str, max_lines: int | None = None, stream: str = "out") -> dict[str, Any]:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"lines": manager.tail(job_id, max_lines, stream)}


class JobRename(BaseModel):
    name: str | None = None


@router.put("/api/jobs/{job_id}", dependencies=[Depends(auth)])
def rename_job(job_id: str, body: JobRename) -> dict[str, Any]:
    try:
        return _with_project_names([manager.rename(job_id, body.name or "")])[0]
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")


@router.get("/api/jobs/{job_id}/copy-prefill", dependencies=[Depends(auth)])
def job_copy_prefill(job_id: str) -> dict[str, Any]:
    """Recipe for copying this job into another project: program + parameters with the
    per-project bits (output/SSD dirs) and data inputs cleared. Seeds the New Run form;
    creates nothing."""
    try:
        return manager.copy_prefill(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")


@router.get("/api/jobs/{job_id}/dependents", dependencies=[Depends(auth)])
def job_dependents(job_id: str) -> dict[str, Any]:
    """The runs this one feeds, transitively — what re-running it would invalidate.

    Read-only: this is what the confirmation dialog is built from, before anything has
    been touched.
    """
    try:
        deps = manager.dependents(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")
    return {"jobs": [cascade.brief(d) for d in deps]}


@router.post("/api/jobs/{job_id}/rerun", dependencies=[Depends(auth)])
async def rerun_job(job_id: str, body: RerunBody = RerunBody()) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # Checked up front, before the cascade: a rerun that is going to be refused anyway
    # must not leave a trail of runs flagged stale for a rebuild that never happened.
    if job.get("active"):
        raise HTTPException(status_code=409, detail="job is already running")
    # Same principle for a 422: dry-validate the stored command before the cascade so a
    # refused rerun leaves no dependents killed or flagged stale.
    try:
        manager.precheck_rerun(job_id)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})
    # Invalidate downstream *before* relaunching: their inputs live in the run folder the
    # rerun is about to delete, and one of them may still be reading it.
    deps = await _apply_cascade([job_id], body)
    try:
        result = await manager.rerun(job_id)
    except JobActive:
        raise HTTPException(status_code=409, detail="job is already running")
    except ProjectNotFound:
        raise HTTPException(status_code=409, detail="the job's project no longer exists")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})
    await _queue_cascade(job.get("project_id"), deps, body)
    return result


@router.post("/api/jobs/{job_id}/continue", dependencies=[Depends(auth)])
async def continue_job(job_id: str, body: RerunBody = RerunBody()) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # Same as rerun: settle whether this can happen at all before invalidating anything.
    # JobManager.continue_job re-checks both — this is only so a refusal leaves no trail.
    if job.get("active"):
        raise HTTPException(status_code=409, detail="job is already running")
    if job.get("stale"):
        raise HTTPException(status_code=409, detail="this run's inputs were re-made after it "
                                                    "finished — re-run it instead of continuing")
    # Also settled up front: whether there is a checkpoint to continue from. Discovering
    # there isn't one *after* the cascade would strand the dependents it invalidated.
    if not adapters.is_resumable(job["program"], job.get("workdir")):
        raise HTTPException(status_code=409, detail="no intermediate checkpoint to continue from")
    # A continue keeps this run's folder, so downstream input *paths* survive — but the
    # outputs behind them are about to be overwritten from a further-trained checkpoint,
    # so anything derived from them is just as out of date.
    deps = await _apply_cascade([job_id], body)
    try:
        result = _with_project_names([await manager.continue_job(job_id)])[0]
    except JobActive:
        raise HTTPException(status_code=409, detail="job is already running")
    except JobNotResumable as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ProjectNotFound:
        raise HTTPException(status_code=409, detail="the job's project no longer exists")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})
    await _queue_cascade(job.get("project_id"), deps, body)
    return result


@router.post("/api/jobs/{job_id}/start", dependencies=[Depends(auth)])
async def start_job(job_id: str) -> dict[str, Any]:
    try:
        return _with_project_names([await manager.start(job_id)])[0]
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")
    except JobActive:
        raise HTTPException(status_code=409, detail="job is already running")
    except JobNotSaved:
        raise HTTPException(status_code=409, detail="only saved jobs can be started")
    # Starting a *stale* saved job rebuilds it from its bindings (jobs.start -> rerun), so
    # it can fail the same ways a rerun can — a source run deleted, or a stored argument
    # the program no longer accepts. Surfaced as a field error instead of a 500.
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})
    except ProjectNotFound:
        raise HTTPException(status_code=409, detail="the job's project no longer exists")


@router.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(auth)])
async def cancel_job(job_id: str) -> dict[str, Any]:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    await manager.cancel(job_id)
    return manager.get(job_id)


@router.delete("/api/jobs/{job_id}", dependencies=[Depends(auth)])
def delete_job(job_id: str) -> dict[str, bool]:
    try:
        manager.delete(job_id)
    except JobActive:
        raise HTTPException(status_code=409, detail="cancel the job before deleting it")
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")
    # Drop the step the job backed from any workflow graph right away, so a canvas
    # reloaded (or polled) before the next full reconcile doesn't show it again.
    wf_manager.forget_job(job_id)
    label_manager.remove_jobs([job_id])   # its job_labels rows must not outlive it
    return {"deleted": True}


@router.websocket("/ws/jobs/{job_id}/log")
async def job_log_ws(websocket: WebSocket, job_id: str) -> None:
    # WebSockets can't send custom headers from the browser, so auth via ?token=.
    if not _ws_authorized(websocket):
        await websocket.close(code=4401)
        return
    job = manager.get(job_id)
    if job is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    # subscribe() registers our queue atomically, and tail_positions() is captured
    # in the same tick: the snapshot then reads the file only up to those offsets,
    # while everything past them arrives through the queue — no chunk duplicated
    # or dropped even though the (multi-MB) file read itself happens in a worker
    # thread. Reading it inline used to freeze the whole event loop for seconds.
    queue, _ = manager.subscribe(job_id)
    cut = manager.tail_positions(job_id)
    try:
        # One snapshot carrying both streams' full replay, so the UI seeds its
        # stdout / stderr tabs; live chunks that follow are tagged per stream.
        def _snapshot() -> dict[str, Any]:
            return {"type": "snapshot", "status": job["status"],
                    "streams": {"out": manager.read_log(job_id, "out", upto=cut.get("out")),
                                "err": manager.read_log(job_id, "err", upto=cut.get("err"))}}

        if not job["active"]:
            # Finished job: replay the full log and close — nothing more will stream.
            manager.unsubscribe(job_id, queue)
            await websocket.send_json(await run_in_threadpool(_snapshot))
            await websocket.send_json({"type": "status", "status": job["status"], "return_code": job["return_code"]})
            return

        await websocket.send_json(await run_in_threadpool(_snapshot))
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
            # Close only on a *terminal* status. 'cancelling' is a transition — the
            # real end state ('cancelled') is published seconds later by the
            # supervisor, and hanging up here left the client stuck on "Cancelling…".
            if message.get("type") == "status" and message.get("status") not in ACTIVE:
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(job_id, queue)

