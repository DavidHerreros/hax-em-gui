"""Infrastructure shared by the route modules.

Everything here used to live inline in the 2,500-line ``app.py``. The route
modules import from here; ``app.py`` keeps only app construction, middleware,
lifespan, and the router include calls.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response, WebSocket
from pydantic import BaseModel

from .. import cascade, projects, updates
from ..config import is_loopback_host, settings
from ..jobs import manager
from ..labels import manager as label_manager
from ..projects import ProjectsRootUnset
from ..workflows import WorkflowError


# -- auth ------------------------------------------------------------------
def auth(authorization: str | None = Header(default=None)) -> None:
    """Header bearer-token check. No token configured => open (dev only)."""
    if not settings.auth_token:
        return
    if authorization != f"Bearer {settings.auth_token}":
        raise HTTPException(status_code=401, detail="invalid or missing token")


def host_allowed(hostport: str | None) -> bool:
    """Reject requests whose Host header we don't recognise (DNS rebinding).

    A hostile page pointed at a domain that resolves to this server becomes
    same-origin with the API — CORS never enters into it — and on an open
    loopback server that is full read + command execution. So: on a loopback
    bind only loopback names are accepted; on a public bind the legitimate name
    can't be inferred, so a list applies only when HAX_GUI_ALLOWED_HOSTS is set
    (auth is always on for public binds, which closes the unauthenticated
    hole). "*" disables the check."""
    if "*" in settings.allowed_hosts:
        return True
    if not hostport:
        return False
    if hostport.startswith("["):                     # [::1]:8000
        host = hostport.split("]", 1)[0] + "]"
    else:
        host = hostport.rsplit(":", 1)[0] if ":" in hostport else hostport
    host = host.strip("[]").lower()
    if any(host == h.strip("[]").lower() for h in settings.allowed_hosts):
        return True
    if settings.is_public_bind:
        return not settings.allowed_hosts
    return is_loopback_host(host)


def ws_authorized(websocket: WebSocket) -> bool:
    """Host allowlist + ?token= (browsers can't set headers on WebSockets)."""
    if not host_allowed(websocket.headers.get("host")):
        return False
    return (not settings.auth_token
            or websocket.query_params.get("token") == settings.auth_token)


# -- reverse proxy ---------------------------------------------------------
# Hop-by-hop headers must not be forwarded through the reverse proxies.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


async def proxy(request: Request, url: str) -> Response:
    """Stream one request through to a local subprocess (TB / viewers).

    Streaming, not buffered: the old handlers awaited the full upstream body
    into memory before answering, so a 512³ map download or a full-resolution
    mesh staged hundreds of MB in this process — per request. Bytes are
    forwarded raw (no decode), so upstream content-encoding/length are
    preserved as-is; the upstream response is closed when the client is done.
    """
    from starlette.background import BackgroundTask
    from starlette.responses import StreamingResponse

    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_BY_HOP and k.lower() != "host"}
    body = await request.body()
    client: httpx.AsyncClient = request.app.state.http
    upstream = await client.send(
        client.build_request(request.method, url, headers=fwd_headers, content=body),
        stream=True,
    )
    out_headers = {k: v for k, v in upstream.headers.items()
                   if k.lower() not in _HOP_BY_HOP}
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=upstream.headers.get("content-type"),
        background=BackgroundTask(upstream.aclose),
    )


# -- shared request models -------------------------------------------------
class JobCreate(BaseModel):
    program: str
    args: dict[str, Any] = {}
    gpu: str | None = None
    # Custom display name. Alongside replace_job_id, omitting it (null) keeps the name the
    # run already has — only an explicit "" clears it, so a client that doesn't carry the
    # name can't wipe it by re-submitting the run.
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
    # "Edit & continue": reuse replace_job_id's run folder (don't wipe it) so the
    # program resumes from the checkpoint it left there, with the edited args.
    resume: bool = False
    # Only meaningful alongside replace_job_id, which re-makes an existing run and so
    # invalidates everything downstream of it — see RerunBody / cascade.py.
    cascade: str = "stale"
    cancel_running: bool = False


class RerunBody(BaseModel):
    """What to do with the runs downstream of the one being re-made.

    Re-running a step deletes and rebuilds its run folder, which is where its dependents
    read their inputs from — so they are invalidated either way (see cascade.py). The
    choice is only whether they are left flagged for the user to re-run when they like
    ('stale'), or handed to the scheduler to re-run themselves in order ('queue').
    """

    cascade: str = "stale"
    # Downstream runs that are still executing are reading the folder about to be
    # deleted. The client re-submits with this set once the user has agreed to stop them.
    cancel_running: bool = False


# -- cascade helpers -------------------------------------------------------
async def apply_cascade(job_ids: list[str], body: RerunBody) -> list[dict[str, Any]]:
    """Invalidate everything downstream of `job_ids`, or refuse if that means killing a run."""
    deps = cascade.dependents_of(job_ids)
    if not deps:
        return []
    try:
        await cascade.invalidate(deps, cancel_running=body.cancel_running)
    except cascade.DependentsRunning as exc:
        raise HTTPException(status_code=409, detail={
            "error": "downstream runs are still executing",
            "running": exc.running,
        })
    return deps


async def queue_cascade(project_id: str | None, deps: list[dict[str, Any]], body: RerunBody) -> None:
    """Hand the invalidated runs to the scheduler, if that is what was asked for."""
    if body.cascade != "queue" or not deps:
        return
    try:
        await cascade.queue_jobs(project_id, [d["id"] for d in deps])
    except WorkflowError as exc:
        # The rerun itself has already been launched, so this is not a failure of the
        # request — only of the automatic follow-up. The steps stay marked stale and can
        # be re-run by hand, which is what the message says.
        raise HTTPException(status_code=409, detail={
            "error": "the downstream steps could not be queued; they are marked stale and "
                     "can be re-run by hand",
            "errors": exc.errors,
        })


# -- lookups & annotations -------------------------------------------------
def project_or_404(project_id: str) -> dict[str, Any]:
    try:
        proj = projects.get(project_id)
    except ProjectsRootUnset:
        proj = None
    if proj is None:
        raise HTTPException(status_code=404, detail=f"unknown project '{project_id}'")
    return proj


def with_project_names(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return attach_labels(jobs)


def attach_labels(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach each job's labels (one batched query)."""
    by_job = label_manager.labels_for_many([j["id"] for j in jobs])
    for job in jobs:
        job["labels"] = by_job.get(job["id"], [])
    return jobs


def has_tb_events(workdir: Path) -> bool:
    if not workdir.is_dir():
        return False
    # Check the run folder and a couple of levels down (cheap, bounded).
    for pattern in ("events.out.tfevents.*", "*/events.out.tfevents.*", "*/*/events.out.tfevents.*"):
        if next(workdir.glob(pattern), None) is not None:
            return True
    return False


# -- update-status cache (lives on app.state; shared by poller + endpoints) --
async def run_update_check(app: FastAPI) -> None:
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


__all__ = [
    "auth", "host_allowed", "ws_authorized", "proxy",
    "JobCreate", "RerunBody", "apply_cascade", "queue_cascade",
    "project_or_404", "with_project_names", "attach_labels", "has_tb_events",
    "run_update_check", "manager",
]
