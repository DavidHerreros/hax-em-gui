"""System routes: health, program catalog, GPUs, updates, files, settings,
git branches, labels, and the in-place server restart."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .. import adapters, branches, gpus, groups, registry, settings_store
from .. import files as filebrowser
from ..branches import BranchError
from ..jobs import manager
from ..labels import manager as label_manager
from ..updater import UpdateBlocked, UpdateBusy, updater
from .shared import auth, run_update_check, ws_authorized as _ws_authorized

router = APIRouter()

@router.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "programs": len(registry.all_programs()), "schema_errors": registry.errors()}

@router.get("/api/programs", dependencies=[Depends(auth)])
def programs() -> list[dict[str, Any]]:
    return registry.list_programs()


@router.get("/api/programs/{name}", dependencies=[Depends(auth)])
def program(name: str) -> dict[str, Any]:
    schema = registry.get_schema(name)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"unknown program '{name}'")
    # The project manager auto-fills the output + SSD scratch dirs, so the form
    # hides those args. `bindable` marks fields that can be fed from a previous
    # job's output, and `producers` lists which programs can feed each kind.
    # `group_order` is the accordion order for the `group` on each field.
    return {
        **schema,
        "group_order": groups.GROUP_ORDER,
        "output_arg": registry.output_arg(name),
        "ssd_arg": registry.ssd_arg(name),
        "bindable": adapters.bindable_fields(name),
        "bindable_multi": adapters.multi_bindable_fields(name),
        "producers": adapters.producers_for(name),
        "produces": adapters.produces(name),
    }


@router.get("/api/gpus", dependencies=[Depends(auth)])
def list_gpus() -> list[dict[str, Any]]:
    return gpus.list_gpus()


@router.get("/api/system/versions", dependencies=[Depends(auth)])
def system_versions() -> dict[str, str]:
    """Currently installed versions of the packages stamped into each job's
    provenance — lets the UI warn when reproducing a run made under different ones."""
    from ..jobs import _pkg_versions
    return _pkg_versions()


@router.get("/api/updates", dependencies=[Depends(auth)])
def get_updates(request: Request) -> dict[str, Any]:
    """Cached update status for hax-em / hax-em-gui (refreshed in the background)."""
    return request.app.state.updates


@router.post("/api/updates/check", dependencies=[Depends(auth)])
async def refresh_updates(request: Request) -> dict[str, Any]:
    """Force an immediate re-check (used by the manual 'Check now' affordance)."""
    if not request.app.state.updates.get("checking"):
        await run_update_check(request.app)
    return request.app.state.updates


class UpdateRun(BaseModel):
    package: str
    # "full" = pull + install deps + restart/finish; "pull" = pull only, then the
    # GUI server closes so the user installs deps and restarts by hand (devel).
    mode: str = "full"


@router.get("/api/updates/run", dependencies=[Depends(auth)])
def update_run_state() -> dict[str, Any]:
    return updater.state()


@router.post("/api/updates/run", dependencies=[Depends(auth)])
async def run_update(body: UpdateRun) -> dict[str, Any]:
    """Install the update for one package (output streams over /ws/updates/log)."""
    try:
        return await updater.run(body.package, body.mode)
    except UpdateBusy:
        raise HTTPException(status_code=409, detail="an update is already running")
    except UpdateBlocked as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.websocket("/ws/updates/log")
async def update_log_ws(websocket: WebSocket) -> None:
    if not _ws_authorized(websocket):
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


@router.get("/api/files", dependencies=[Depends(auth)])
def browse(path: str | None = None, fallback: bool = False) -> dict[str, Any]:
    try:
        return filebrowser.list_dir(path, fallback=fallback)
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


class DirCreate(BaseModel):
    path: str | None = None   # parent directory; defaults to the first file root
    name: str


@router.post("/api/files/mkdir", dependencies=[Depends(auth)])
def make_directory(body: DirCreate) -> dict[str, Any]:
    try:
        return filebrowser.make_dir(body.path, body.name)
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except filebrowser.InvalidName as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileExistsError:
        raise HTTPException(status_code=409, detail=f"already exists: {body.name}")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# -- settings --------------------------------------------------------------
class SettingsUpdate(BaseModel):
    projects_root: str | None = None
    ssd_scratch_root: str | None = None
    # Device the landscape viewer loads the network and decodes maps on: a GPU index
    # ("0"), "cpu", or "" to be asked at launch time (see _viewer_gpu).
    viewer_gpu: str | None = None
    # Jobs admitted per GPU at once (admission control; 0 disables the queue).
    max_jobs_per_gpu: int | None = None


@router.get("/api/settings", dependencies=[Depends(auth)])
def get_settings() -> dict[str, Any]:
    return settings_store.load()


@router.put("/api/settings", dependencies=[Depends(auth)])
def put_settings(body: SettingsUpdate) -> dict[str, Any]:
    return settings_store.update(body.model_dump(exclude_none=True))


# Identifies this server process. The SPA scopes "remind me later" to it, so the
# setup notice returns after a restart (or in a fresh browser session), but not
# on a plain page reload.
_SERVER_ID = uuid4().hex


@router.get("/api/setup-notice", dependencies=[Depends(auth)])
def setup_notice() -> dict[str, Any]:
    return {**settings_store.setup_status(), "server_id": _SERVER_ID}


@router.post("/api/setup-notice/silence", dependencies=[Depends(auth)])
def silence_setup_notice() -> dict[str, Any]:
    """Stop nagging about the roots that are currently unset ("close forever")."""
    return {**settings_store.silence_setup_notice(), "server_id": _SERVER_ID}
# -- developer / git branches (editable installs only) ---------------------
class BranchSwitch(BaseModel):
    package: str
    branch: str


class PackageRef(BaseModel):
    package: str


class BranchSource(BaseModel):
    package: str
    path: str | None = None  # empty/None clears the configured source clone


@router.get("/api/branches", dependencies=[Depends(auth)])
async def list_branches() -> list[dict[str, Any]]:
    """Branch status for every watched package (a stub where no checkout exists)."""
    return await asyncio.to_thread(branches.list_all)


@router.post("/api/branches/source", dependencies=[Depends(auth)])
async def set_branch_source(body: BranchSource) -> dict[str, Any]:
    """Point a package at a source clone the install itself doesn't reference."""
    try:
        return await asyncio.to_thread(branches.set_source, body.package, body.path)
    except BranchError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/api/branches/fetch", dependencies=[Depends(auth)])
async def fetch_branches(body: PackageRef) -> dict[str, Any]:
    """Refresh remote-tracking refs for one package, then return fresh status."""
    info = await asyncio.to_thread(branches.fetch, body.package)
    if info is None:
        raise HTTPException(status_code=404, detail=f"'{body.package}' is not a development install")
    return info


@router.post("/api/branches/switch", dependencies=[Depends(auth)])
async def switch_branch(body: BranchSwitch, request: Request) -> dict[str, Any]:
    """Check out a branch (refused on a dirty tree or an unknown branch)."""
    if manager.count_active() > 0:
        raise HTTPException(status_code=409, detail="a job is still running — wait for it to finish")
    try:
        result = await asyncio.to_thread(branches.switch, body.package, body.branch)
    except BranchError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    # The new branch may sit at a different version vs its upstream — re-probe.
    await run_update_check(request.app)
    return result


@router.post("/api/server/restart", dependencies=[Depends(auth)])
def restart_server(request: Request) -> dict[str, Any]:
    """Restart the server in place (e.g. after switching the GUI's own branch).

    Reuses the same detached handshake as a GUI self-update; the client then
    health-polls and reloads. False when launched outside ``hax-gui start``."""
    from .. import cli

    # Preserve the Ollama we started across the restart (the new process reclaims
    # it via the pidfile) instead of killing and reloading the model.
    request.app.state.keep_ollama = True
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


@router.get("/api/labels", dependencies=[Depends(auth)])
def list_labels() -> list[dict[str, Any]]:
    return label_manager.list()


@router.post("/api/labels", dependencies=[Depends(auth)])
def create_label(body: LabelBody) -> dict[str, Any]:
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="label name required")
    return label_manager.create(body.name, body.color)


@router.put("/api/labels/{lid}", dependencies=[Depends(auth)])
def update_label(lid: str, body: LabelUpdate) -> dict[str, Any]:
    try:
        return label_manager.update(lid, name=body.name, color=body.color)
    except KeyError:
        raise HTTPException(status_code=404, detail="label not found")


@router.delete("/api/labels/{lid}", dependencies=[Depends(auth)])
def delete_label(lid: str) -> dict[str, bool]:
    label_manager.delete(lid)
    return {"deleted": True}


@router.put("/api/jobs/{job_id}/labels", dependencies=[Depends(auth)])
def set_job_labels(job_id: str, body: JobLabels) -> dict[str, Any]:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"labels": label_manager.set_job_labels(job_id, body.labels)}



@router.websocket("/ws/events")
async def events_ws(websocket: WebSocket) -> None:
    """Global server-push channel: one JSON message per job status transition.

    The SPA opens one of these per tab; views kick their pollers on receipt
    (poll stays as the fallback), and the notifications module watches for
    terminal states. See hax_gui/events.py."""
    from .. import events

    if not _ws_authorized(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    queue = events.subscribe()
    try:
        shutdown = websocket.app.state.shutdown
        while True:
            getter = asyncio.ensure_future(queue.get())
            stopper = asyncio.ensure_future(shutdown.wait())
            done, pending = await asyncio.wait({getter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if stopper in done:
                break
            await websocket.send_json(getter.result())
    except WebSocketDisconnect:
        pass
    finally:
        events.unsubscribe(queue)
