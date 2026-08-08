"""FastAPI application assembly: middleware, lifespan, routers, static SPA.

Run (inside the hax conda env):
    uvicorn hax_gui.app:app --host 0.0.0.0 --port 8000

The endpoints themselves live in :mod:`hax_gui.routes` — one module per area
(system, projects, jobs, workflows, metrics, viewers, assistant), with the
shared plumbing (auth, Host allowlist, streaming proxy, cascade helpers) in
:mod:`hax_gui.routes.shared`.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import assistant as assistant_bridge
from . import chats as chat_store
from . import children
from . import ollama as ollama_manager_mod
from .config import settings
from .jobs import manager
from .labels import manager as label_manager
from .md_viewer import md_viewer_manager
from .mrc_viewer import mrc_viewer_manager
from .routes import (
    assistant as assistant_routes,
    jobs as job_routes,
    metrics as metrics_routes,
    projects as project_routes,
    shared,
    system as system_routes,
    viewers as viewer_routes,
    workflows as workflow_routes,
)
from .tensorboard import tb_manager
from .viewer import viewer_manager
from .workflows import manager as wf_manager


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    await _on_startup(app)
    try:
        yield
    finally:
        await _on_shutdown(app)


app = FastAPI(title="Hax GUI", version="0.1.0", lifespan=_lifespan)
# Compress responses when the browser accepts it (mesh/grid binaries and JSON
# job lists shrink a lot); already-encoded upstream bodies are left alone
# (starlette's guard), and small responses aren't worth the header.
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Defence in depth for the SPA (the app builds its DOM via textContent, but one
# missed `html:` prop is all it takes — see the TB add_text legend fix). Inline
# <script> is forbidden ('self' only); styles are applied via CSSOM, which CSP
# does not restrict; images use blob:/data: object URLs; WebSockets are
# same-origin. Proxied apps (TensorBoard, the viewer subprocesses) ship their
# own inline scripts, so the header is not applied under their prefixes.
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' blob: data:; connect-src 'self' ws: wss:; "
    "frame-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'"
)
_CSP_EXEMPT = ("/tb/", "/viewer/", "/mrcviewer/", "/mdviewer/")


# --- auth-by-default -------------------------------------------------------
# The job API is arbitrary command execution as the server user, so an open
# server is only tolerable when nothing but this machine can reach it. On a
# public bind with no HAX_GUI_TOKEN the app adopts a persisted token instead of
# starting open (hax-gui start prints where it lives).
_auth_note = settings.ensure_auth_for_bind()
if _auth_note:
    print(f"[hax-gui] {_auth_note}", flush=True)
print("[hax-gui] auth: " + ("token required" if settings.auth_token else
      "OPEN (no token; loopback only — set HAX_GUI_TOKEN before sharing this machine)"),
      flush=True)


@app.middleware("http")
async def host_allowlist(request: Request, call_next):
    if not shared.host_allowed(request.headers.get("host")):
        return Response("unrecognized Host header", status_code=421, media_type="text/plain")
    return await call_next(request)


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    """Force the browser to revalidate static assets so edits show up without a
    hard refresh (StaticFiles still answers 304 when a file is unchanged)."""
    response = await call_next(request)
    if not request.url.path.startswith(("/api", "/ws", "/tb")):
        response.headers["Cache-Control"] = "no-cache"
    if not request.url.path.startswith(_CSP_EXEMPT):
        response.headers["Content-Security-Policy"] = _CSP
    return response


# How often the background poller re-checks PyPI / git for newer versions. The
# result is cached on app.state so page loads read it instantly (never block on
# the network); the first check fires shortly after startup.
_UPDATE_POLL_SECONDS = 6 * 60 * 60


async def _on_startup(app: FastAPI) -> None:
    manager.init()
    label_manager.init()
    wf_manager.init()
    chat_store.manager.init()
    app.state.http = httpx.AsyncClient(timeout=None)
    app.state.shutdown = asyncio.Event()
    app.state.keep_ollama = False  # set true across an in-place restart (see restart_server)
    app.state.updates = {"checked_at": None, "checking": False, "packages": []}
    app.state.update_poller = asyncio.create_task(_poll_updates())
    # Jobs outlive this process now, so a restart does not end them — pick up the ones
    # left running by the previous server and resume streaming their logs. Workflow
    # schedulers, being asyncio tasks rather than processes, cannot outlive it: restart
    # the ones whose run is still genuinely in flight (workflows._plan_resume decides;
    # the rest are parked for the user to confirm, since resuming them would launch work).
    app.state.job_reattach = asyncio.create_task(manager.reattach())
    app.state.wf_resume = asyncio.create_task(wf_manager.resume_all())
    # TensorBoard/viewer children a crashed server left behind keep holding their
    # ports (and, for landscape viewers, GPU memory) — kill them now, off the loop.
    app.state.child_reap = asyncio.create_task(asyncio.to_thread(children.reap_orphans))
    app.state.viewer_reaper = asyncio.create_task(_reap_idle_viewers())
    # Assistant warmup runs in the background: the old sync startup handler blocked the
    # accept loop for up to 20 s polling Ollama — and built the docs index (via
    # assistant status) BEFORE starting Ollama, leaving the index without
    # embeddings for the life of the process.
    app.state.ollama = {"managed": False}
    app.state.assistant_warmup = asyncio.create_task(_warm_assistant(app))


async def _warm_assistant(app: FastAPI) -> None:
    """Bring up Ollama (when managed), then build the assistant's doc index —
    in that order, so the index embeds against a live Ollama."""
    import os
    if os.environ.get("HAX_GUI_ASSISTANT_WARMUP", "1") in ("0", "false", "no"):
        return   # tests / minimal deployments: skip Ollama and the index build
    if not assistant_bridge.available():
        return
    if settings.manage_ollama:
        try:
            url = assistant_bridge.base_url()   # config only; must not build the index
            if url:
                app.state.ollama = await asyncio.to_thread(
                    ollama_manager_mod.manager.ensure_started, url)
        except Exception:  # noqa: BLE001 - never fail startup on Ollama
            app.state.ollama = {"managed": False, "reason": "startup error"}
    try:
        # Constructs the session, which builds the docs index with embeddings.
        await asyncio.to_thread(assistant_bridge.status)
    except Exception:  # noqa: BLE001 - the assistant stays optional
        pass


async def _on_shutdown(app: FastAPI) -> None:
    app.state.shutdown.set()  # unblock idle log WebSockets so the server can exit
    for name in ("update_poller", "viewer_reaper"):
        task = getattr(app.state, name, None)
        if task is not None:
            task.cancel()
    tb_manager.shutdown()
    viewer_manager.shutdown()
    mrc_viewer_manager.shutdown()
    md_viewer_manager.shutdown()
    # Stop the Ollama we started — but not across an in-place restart, so the new
    # process can reclaim it (avoids a costly model reload). A real shutdown stops it.
    if not getattr(app.state, "keep_ollama", False):
        try:
            ollama_manager_mod.manager.stop_if_owned()
        except Exception:  # noqa: BLE001
            pass
    await app.state.http.aclose()


async def _reap_idle_viewers() -> None:
    """Stop viewer sessions nobody has touched for `viewer_idle_minutes`.

    An abandoned landscape viewer holds GPU memory for as long as it lives; the
    map/metadata viewers hold a port and their loaded data. Every proxied request
    counts as activity (see the `touch` calls in the proxy routes), so an open,
    in-use viewer tab is never reaped — only sessions whose tab was closed or
    forgotten. TensorBoard is deliberately exempt: it follows a training run the
    user may check hours apart, and holds no GPU."""
    if settings.viewer_idle_minutes <= 0:
        return
    max_idle = settings.viewer_idle_minutes * 60
    while not app.state.shutdown.is_set():
        try:
            await asyncio.wait_for(app.state.shutdown.wait(), timeout=60)
            break     # shutdown was signalled
        except asyncio.TimeoutError:
            pass
        for mgr in (viewer_manager, mrc_viewer_manager, md_viewer_manager):
            try:
                mgr.reap_idle(max_idle)
            except Exception:  # noqa: BLE001 - the reaper must outlive one bad pass
                pass


async def _poll_updates() -> None:
    """Background loop: check on startup, then every _UPDATE_POLL_SECONDS."""
    while not app.state.shutdown.is_set():
        await shared.run_update_check(app)
        try:
            await asyncio.wait_for(app.state.shutdown.wait(), timeout=_UPDATE_POLL_SECONDS)
        except asyncio.TimeoutError:
            continue
        break  # shutdown was signalled


# -- routers ---------------------------------------------------------------
app.include_router(system_routes.router)
app.include_router(assistant_routes.router)
app.include_router(project_routes.router)
app.include_router(job_routes.router)
app.include_router(metrics_routes.router)
app.include_router(viewer_routes.router)
app.include_router(workflow_routes.router)


# -- static SPA (mounted last so /api and /ws take precedence) -------------
# The frontend is plain ES modules + CSS (no build step), so we serve the
# directory directly. Same-origin => no CORS/proxy needed for production use.
def mount_spa() -> None:
    # The canvas plotter and the latent scatter live in the viewer app's static dir.
    # The SPA's metrics view imports the very same modules from /shared, rather than
    # keeping a forked copy in sync by hand.
    shared_dir = Path(__file__).resolve().parent / "viewer_app" / "static"
    if shared_dir.is_dir():
        app.mount("/shared", StaticFiles(directory=str(shared_dir)), name="shared")

    frontend = settings.frontend_dir
    if (frontend / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="spa")


mount_spa()
