"""FastAPI application: program catalog, job execution, live logs, file browser.

Run (inside the hax conda env):
    uvicorn hax_gui.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import adapters
from . import assistant as assistant_bridge
from . import assistant_actions
from . import chats as chat_store
from . import metrics as metrics_reader
from . import ollama as ollama_manager_mod
from . import files as filebrowser
from . import branches, cascade, gpus, groups, projects, registry, settings_store, updates, usage
from .branches import BranchError
from .updater import UpdateBlocked, UpdateBusy, updater
from .command import ValidationError
from .config import is_loopback_host, settings
from .jobs import ACTIVE, JobActive, JobNotFound, JobNotResumable, JobNotSaved, ProjectNotFound, manager
from .labels import manager as label_manager
from .workflows import StepsRunning, WorkflowError, manager as wf_manager
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


def _host_allowed(hostport: str | None) -> bool:
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


@app.middleware("http")
async def host_allowlist(request: Request, call_next):
    if not _host_allowed(request.headers.get("host")):
        return Response("unrecognized Host header", status_code=421, media_type="text/plain")
    return await call_next(request)


def _ws_authorized(websocket: WebSocket) -> bool:
    """Host allowlist + ?token= (browsers can't set headers on WebSockets)."""
    if not _host_allowed(websocket.headers.get("host")):
        return False
    return (not settings.auth_token
            or websocket.query_params.get("token") == settings.auth_token)


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


@app.on_event("startup")
def _startup() -> None:
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

    # Optionally bring up a local Ollama for the assistant (only if not already
    # running); shutdown stops only an instance we started. See ollama.py.
    app.state.ollama = {"managed": False}
    if settings.manage_ollama and assistant_bridge.available():
        try:
            base_url = assistant_bridge.status().get("base_url", "")
            if base_url:
                app.state.ollama = ollama_manager_mod.manager.ensure_started(base_url)
        except Exception:  # noqa: BLE001 - never block startup on Ollama
            app.state.ollama = {"managed": False, "reason": "startup error"}


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
    # Stop the Ollama we started — but not across an in-place restart, so the new
    # process can reclaim it (avoids a costly model reload). A real shutdown stops it.
    if not getattr(app.state, "keep_ollama", False):
        try:
            ollama_manager_mod.manager.stop_if_owned()
        except Exception:  # noqa: BLE001
            pass
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


# -- catalog ---------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "programs": len(registry.all_programs()), "schema_errors": registry.errors()}


# -- optional AI assistant -------------------------------------------------
@app.get("/api/assistant/status", dependencies=[Depends(auth)])
def assistant_status() -> dict[str, Any]:
    """Availability + config of the optional hax-em-assistant. Always present so
    the SPA can decide whether to show the chat panel; reports available=False
    (never errors) when the package is not installed."""
    st = assistant_bridge.status()
    st["actions_enabled"] = assistant_actions.actions_enabled()
    return st


@app.post("/api/assistant/reindex-docs", dependencies=[Depends(auth)])
def assistant_reindex_docs(force: bool = True) -> dict[str, Any]:
    """Rebuild the assistant's docs vector index from the current hax-em-docs
    checkout, so edits to the docs are picked up without restarting the GUI.
    With ``force=false`` it only rebuilds when the docs content hash changed."""
    if not assistant_bridge.available():
        raise HTTPException(status_code=400, detail="assistant not installed")
    return assistant_bridge.reindex_docs(force=force)


class JobProposal(BaseModel):
    program: str
    path: str | None = None


@app.post("/api/assistant/actions/preview-job", dependencies=[Depends(auth)])
def assistant_preview_job(body: JobProposal) -> dict[str, Any]:
    """Deterministically expand an assistant job proposal (program + data path)
    into a draft spec (registry defaults + path→input arg) plus facts about the
    path, so the confirmation card can show exactly what will be created. Builds
    nothing and launches nothing."""
    if not assistant_actions.actions_enabled():
        raise HTTPException(status_code=403, detail="assistant actions are disabled")
    spec = assistant_actions.build_draft_spec(body.program, body.path)
    if spec.get("error"):
        raise HTTPException(status_code=422, detail=spec["error"])
    inspect = assistant_actions.inspect_path(body.path) if body.path else None
    return {"spec": spec, "inspect": inspect}


@app.get("/api/assistant/actions/prepare-motion/{job_id}", dependencies=[Depends(auth)])
def assistant_prepare_motion_plan(job_id: str) -> dict[str, Any]:
    """Read-only plan for preparing a run's deformation field (per-domain motion),
    plus the visible GPUs — shown in the confirmation before any GPU run."""
    if not assistant_bridge.available():
        raise HTTPException(status_code=400, detail="assistant not installed")
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    art = _resolve_run_artifacts(job)
    if not art["run_dir"]:
        raise HTTPException(status_code=422, detail="run has no folder to analyse")
    return {"plan": assistant_bridge.plan_prepare_motion(art["run_dir"]),
            "gpus": assistant_bridge.list_gpus()}


@app.websocket("/ws/assistant/prepare-motion")
async def assistant_prepare_motion_ws(websocket: WebSocket) -> None:
    """Run the (confirmed) prepare-motion pipeline for a job, streaming progress
    lines. Client sends ``{job_id, gpu}``; server streams ``{status}`` then a final
    ``{done, deformation_dir}`` or ``{error}``."""
    if not _ws_authorized(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    if not assistant_bridge.available():
        await websocket.send_json({"error": "assistant not installed"})
        await websocket.close()
        return
    loop = asyncio.get_running_loop()
    try:
        req = await websocket.receive_json()
        job = manager.get(req.get("job_id"))
        if job is None:
            await websocket.send_json({"error": "job not found"})
            return
        art = _resolve_run_artifacts(job)
        if not art["run_dir"]:
            await websocket.send_json({"error": "run has no folder to analyse"})
            return
        gpu = req.get("gpu", 0)

        out: asyncio.Queue = asyncio.Queue()

        def _progress(line: str) -> None:
            loop.call_soon_threadsafe(out.put_nowait, ("status", line))

        def _worker() -> None:
            try:
                d = assistant_bridge.run_prepare_motion(art["run_dir"], gpu, on_progress=_progress)
                loop.call_soon_threadsafe(out.put_nowait, ("done", d))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(out.put_nowait, ("error", str(exc)))

        loop.run_in_executor(None, _worker)
        while True:
            kind, val = await out.get()
            if kind == "status":
                await websocket.send_json({"status": val})
            elif kind == "done":
                await websocket.send_json({"done": True, "deformation_dir": val})
                break
            else:
                await websocket.send_json({"error": val})
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


@app.get("/api/assistant/run-preview/{job_id}", dependencies=[Depends(auth)])
def assistant_run_preview(job_id: str) -> dict[str, Any]:
    """Which artifacts a job exposes to the assistant (metrics/latents/volumes),
    so the chat's attach-run picker can show what a summary will be able to cover."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    art = _resolve_run_artifacts(job)
    return {
        "job_id": job_id,
        "label": art["label"],
        "has_metrics": art["has_metrics"],
        "has_latents": art["latents"] is not None,
        "has_volumes": art["volumes"] is not None,
        "latents_from": art["latents_from"],  # sibling run name, if borrowed
    }


@app.post("/api/assistant/run-report/{job_id}", dependencies=[Depends(auth)])
def assistant_run_report(job_id: str, narrate: bool = True) -> FileResponse:
    """Generate a PDF run report (deterministic figures + measured values, plus an
    optional AI-generated discussion) for a job's run and return it for download.
    Blocking (figures + optional narration); FastAPI runs it in a threadpool."""
    if not assistant_bridge.available():
        raise HTTPException(status_code=400, detail="assistant not installed")
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    art = _resolve_run_artifacts(job)
    if not (art["run_dir"] or art["latents"] or art["volumes"]):
        raise HTTPException(status_code=422, detail="run has no readable artifacts to report on")
    reports_dir = settings.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{job_id}.pdf"
    program = job.get("program") or "run"
    subtitle = " · ".join(x for x in (job.get("project_id"), program, job.get("name")) if x)
    try:
        assistant_bridge.generate_report(
            str(out), run_dir=art["run_dir"], latents=art["latents"], volumes=art["volumes"],
            title=f"{program} run report", subtitle=subtitle, narrate=narrate)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the client
        raise HTTPException(status_code=500, detail=f"report generation failed: {exc}")
    return FileResponse(str(out), media_type="application/pdf",
                        filename=f"{program}_{job_id}_report.pdf")


@app.get("/api/ollama/status", dependencies=[Depends(auth)])
def ollama_status() -> dict[str, Any]:
    """Whether a local Ollama is running and whether the GUI is managing it."""
    base_url = ""
    if assistant_bridge.available():
        try:
            base_url = assistant_bridge.status().get("base_url", "")
        except Exception:  # noqa: BLE001
            base_url = ""
    out: dict[str, Any] = {"manage_enabled": settings.manage_ollama,
                           "startup": getattr(app.state, "ollama", {})}
    if base_url:
        out.update(ollama_manager_mod.manager.status(base_url))
    return out


@app.websocket("/ws/assistant/chat")
async def assistant_chat_ws(websocket: WebSocket) -> None:
    if not _ws_authorized(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    if not assistant_bridge.available():
        await websocket.send_json({"error": "assistant not installed"})
        await websocket.close()
        return

    loop = asyncio.get_running_loop()

    # A single long-lived receiver owns websocket.receive_json(); every client
    # message lands in `inbox`. This matters because real uvicorn/websockets does
    # not tolerate a receive() being cancelled and then re-issued (which a
    # per-turn stop-watcher would do) — the second turn's message would be lost.
    # Here the only thing ever cancelled is inbox.get(), which is always safe.
    # A None sentinel in the inbox signals disconnect.
    inbox: asyncio.Queue = asyncio.Queue()

    async def receiver() -> None:
        try:
            while True:
                inbox.put_nowait(await websocket.receive_json())
        except Exception:  # noqa: BLE001 - disconnect or malformed frame
            inbox.put_nowait(None)

    recv = asyncio.ensure_future(receiver())
    try:
        while True:
            req = await inbox.get()
            if req is None:  # client disconnected
                break
            if not isinstance(req, dict) or "messages" not in req:
                continue  # stray message with no active turn (e.g. a late stop)
            messages = req.get("messages", [])
            grounded = bool(req.get("grounded", True))
            # Optional attached run: resolve the job id to artifact paths server-side
            # (the client never handles filesystem paths).
            run = None
            run_job_id = req.get("run_job_id")
            if run_job_id:
                job = manager.get(run_job_id)
                if job is not None:
                    run = _resolve_run_artifacts(job)
                    run["job_id"] = run_job_id   # so a proposed action can target it
            allow_actions = assistant_actions.actions_enabled()

            # Run the blocking LLM stream off the event loop, funnel events back.
            # A cancel Event lets a client "stop" actually halt generation (the
            # worker closes the generator, which tears down the model request).
            out: asyncio.Queue = asyncio.Queue()
            cancel = threading.Event()

            def _worker(messages=messages, grounded=grounded, run=run,
                        allow_actions=allow_actions, cancel=cancel, out=out) -> None:
                gen = assistant_bridge.stream_events(messages, grounded=grounded, run=run,
                                                     allow_actions=allow_actions)
                try:
                    for ev in gen:
                        if cancel.is_set():
                            break
                        etype = ev.get("type")
                        if etype == "status":
                            payload = ("status", ev.get("text"))
                        elif etype == "action":
                            payload = ("action", ev.get("action"))
                        else:
                            payload = ("token", ev.get("delta"))
                        loop.call_soon_threadsafe(out.put_nowait, payload)
                except Exception as exc:  # noqa: BLE001 - report, keep socket alive
                    loop.call_soon_threadsafe(out.put_nowait, ("error", str(exc)))
                finally:
                    gen.close()  # stop the underlying model stream promptly on cancel
                    loop.call_soon_threadsafe(out.put_nowait, ("done", None))

            loop.run_in_executor(None, _worker)

            # Forward events while watching the inbox for a client stop message
            # (any inbound message during a turn = stop). Keep draining after a
            # stop until the worker reports done, but stop forwarding tokens.
            get_task = asyncio.ensure_future(out.get())
            inbox_task = asyncio.ensure_future(inbox.get())
            stopped = False
            disconnected = False
            try:
                while True:
                    waiters = {get_task} | ({inbox_task} if inbox_task else set())
                    done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                    if inbox_task and inbox_task in done:
                        msg = inbox_task.result()
                        cancel.set()
                        stopped = True
                        if msg is None:            # disconnect mid-turn
                            disconnected = True
                            inbox_task = None      # stop re-arming; drain to done then exit
                        else:
                            inbox_task = asyncio.ensure_future(inbox.get())
                    if get_task in done:
                        kind, value = get_task.result()
                        if kind == "done":
                            break
                        if not stopped:
                            await websocket.send_json({kind: value})
                        get_task = asyncio.ensure_future(out.get())
            finally:
                get_task.cancel()
                if inbox_task:
                    inbox_task.cancel()
            if disconnected:
                break
            await websocket.send_json({"done": True})
    except WebSocketDisconnect:
        pass
    finally:
        recv.cancel()


# -- assistant chat history (plain data; exists even without the assistant) --
class ChatCreate(BaseModel):
    project_id: str | None = None
    title: str | None = None


class ChatUpdate(BaseModel):
    title: str | None = None
    project_id: str | None = None
    messages: list | None = None


@app.get("/api/chats", dependencies=[Depends(auth)])
def list_chats(project_id: str | None = None) -> list[dict[str, Any]]:
    return chat_store.manager.list(project_id)


@app.post("/api/chats", dependencies=[Depends(auth)])
def create_chat(body: ChatCreate) -> dict[str, Any]:
    return chat_store.manager.create(body.project_id, body.title)


@app.get("/api/chats/{cid}", dependencies=[Depends(auth)])
def get_chat(cid: str) -> dict[str, Any]:
    chat = chat_store.manager.get(cid)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return chat


@app.put("/api/chats/{cid}", dependencies=[Depends(auth)])
def update_chat(cid: str, body: ChatUpdate) -> dict[str, Any]:
    chat = chat_store.manager.update(
        cid,
        title=body.title,
        project_id=body.project_id,
        project_id_set="project_id" in body.model_fields_set,
        messages=body.messages,
    )
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return chat


@app.delete("/api/chats/{cid}", dependencies=[Depends(auth)])
def delete_chat(cid: str) -> dict[str, Any]:
    if not chat_store.manager.delete(cid):
        raise HTTPException(status_code=404, detail="chat not found")
    return {"ok": True}


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


@app.get("/api/files", dependencies=[Depends(auth)])
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


@app.post("/api/files/mkdir", dependencies=[Depends(auth)])
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


@app.get("/api/settings", dependencies=[Depends(auth)])
def get_settings() -> dict[str, Any]:
    return settings_store.load()


@app.put("/api/settings", dependencies=[Depends(auth)])
def put_settings(body: SettingsUpdate) -> dict[str, Any]:
    return settings_store.update(body.model_dump(exclude_none=True))


# Identifies this server process. The SPA scopes "remind me later" to it, so the
# setup notice returns after a restart (or in a fresh browser session), but not
# on a plain page reload.
_SERVER_ID = uuid4().hex


@app.get("/api/setup-notice", dependencies=[Depends(auth)])
def setup_notice() -> dict[str, Any]:
    return {**settings_store.setup_status(), "server_id": _SERVER_ID}


@app.post("/api/setup-notice/silence", dependencies=[Depends(auth)])
def silence_setup_notice() -> dict[str, Any]:
    """Stop nagging about the roots that are currently unset ("close forever")."""
    return {**settings_store.silence_setup_notice(), "server_id": _SERVER_ID}


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


@app.get("/api/projects/usage", dependencies=[Depends(auth)])
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


@app.get("/api/projects/{project_id}/usage", dependencies=[Depends(auth)])
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


@app.post("/api/projects/{project_id}/relink", dependencies=[Depends(auth)])
def relink_project(project_id: str) -> dict[str, Any]:
    """Recover input bindings that an edit-and-rerun flattened into literal paths, so
    the workflow edges they back reappear. Returns what was reconnected."""
    _project_or_404(project_id)
    restored = manager.relink_project(project_id)
    return {"relinked": restored, "count": len(restored)}


@app.get("/api/projects/{project_id}/sources", dependencies=[Depends(auth)])
def project_sources(project_id: str, kind: str) -> list[dict[str, Any]]:
    """Jobs in this project whose output can feed a `kind` input (for chaining)."""
    _project_or_404(project_id)
    return adapters.compatible_sources(kind, manager.list_for_project(project_id))


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


async def _apply_cascade(job_ids: list[str], body: RerunBody) -> list[dict[str, Any]]:
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


async def _queue_cascade(project_id: str | None, deps: list[dict[str, Any]], body: RerunBody) -> None:
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
    # Editing a run in place re-makes its outputs exactly as a rerun does, so the same
    # invalidation applies to everything downstream. Done before the create call, which
    # deletes the folder those dependents read from.
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


@app.get("/api/jobs/{job_id}/log", dependencies=[Depends(auth)])
def job_log(job_id: str, max_lines: int | None = None, stream: str = "out") -> dict[str, Any]:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"lines": manager.tail(job_id, max_lines, stream)}


class JobRename(BaseModel):
    name: str | None = None


@app.put("/api/jobs/{job_id}", dependencies=[Depends(auth)])
def rename_job(job_id: str, body: JobRename) -> dict[str, Any]:
    try:
        return _with_project_names([manager.rename(job_id, body.name or "")])[0]
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")


@app.get("/api/jobs/{job_id}/copy-prefill", dependencies=[Depends(auth)])
def job_copy_prefill(job_id: str) -> dict[str, Any]:
    """Recipe for copying this job into another project: program + parameters with the
    per-project bits (output/SSD dirs) and data inputs cleared. Seeds the New Run form;
    creates nothing."""
    try:
        return manager.copy_prefill(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="job not found")


@app.get("/api/jobs/{job_id}/dependents", dependencies=[Depends(auth)])
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


@app.post("/api/jobs/{job_id}/rerun", dependencies=[Depends(auth)])
async def rerun_job(job_id: str, body: RerunBody = RerunBody()) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # Checked up front, before the cascade: a rerun that is going to be refused anyway
    # must not leave a trail of runs flagged stale for a rebuild that never happened.
    if job.get("active"):
        raise HTTPException(status_code=409, detail="job is already running")
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


@app.post("/api/jobs/{job_id}/continue", dependencies=[Depends(auth)])
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
    # Starting a *stale* saved job rebuilds it from its bindings (jobs.start -> rerun), so
    # it can fail the same ways a rerun can — a source run deleted, or a stored argument
    # the program no longer accepts. Surfaced as a field error instead of a 500.
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})
    except ProjectNotFound:
        raise HTTPException(status_code=409, detail="the job's project no longer exists")


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
    # Drop the step the job backed from any workflow graph right away, so a canvas
    # reloaded (or polled) before the next full reconcile doesn't show it again.
    wf_manager.forget_job(job_id)
    return {"deleted": True}


@app.websocket("/ws/jobs/{job_id}/log")
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

    # Preserve the Ollama we started across the restart (the new process reclaims
    # it via the pidfile) instead of killing and reloading the model.
    app.state.keep_ollama = True
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
            # started_at/finished_at/active let the picker date the run itself rather than
            # the row: created_at survives a rerun, so alone it dates a run that is gone.
            out.append({k: job[k] for k in ("id", "program", "name", "workdir", "created_at",
                                            "started_at", "finished_at", "active", "status", "project_id")})
    return _with_project_names(out)


@app.delete("/api/tensorboard/{tb_id}", dependencies=[Depends(auth)])
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


@app.get("/api/metrics/manifest", dependencies=[Depends(auth)])
async def metrics_manifest(logdir: str) -> dict[str, Any]:
    """Which tags a run logged, of what kind, in which sub-runs, at which steps."""
    return await run_in_threadpool(_metrics_call, metrics_reader.manifest, logdir)


@app.get("/api/metrics/scalars", dependencies=[Depends(auth)])
async def metrics_scalars(logdir: str, tags: str | None = None) -> dict[str, Any]:
    want = [t for t in tags.split("\n") if t] if tags else None
    return await run_in_threadpool(_metrics_call, metrics_reader.scalars, logdir, want)


@app.get("/api/metrics/image", dependencies=[Depends(auth)])
async def metrics_image(logdir: str, run: str, tag: str, frame: int | None = None) -> Response:
    png = await run_in_threadpool(_metrics_call, metrics_reader.image, logdir, run, tag, frame)
    return Response(content=png, media_type="image/png")


@app.get("/api/metrics/embedding", dependencies=[Depends(auth)])
async def metrics_embedding(logdir: str, step: int, tag: str | None = None) -> dict[str, Any]:
    """One epoch of the projector's latent space. Re-read from disk every call, so
    epochs written after the page loaded just show up."""
    return await run_in_threadpool(_metrics_call, metrics_reader.embedding, logdir, step, tag)


@app.get("/api/metrics/sprite", dependencies=[Depends(auth)])
async def metrics_sprite(logdir: str, step: int, tag: str | None = None) -> Response:
    png = await run_in_threadpool(_metrics_call, metrics_reader.sprite, logdir, step, tag)
    return Response(content=png, media_type="image/png")


@app.delete("/api/metrics/cache", dependencies=[Depends(auth)])
async def metrics_drop(logdir: str) -> dict[str, bool]:
    """Release a log dir's accumulators (its image reservoir can be sizeable)."""
    return {"dropped": await run_in_threadpool(_metrics_call, metrics_reader.drop, logdir)}


@app.post("/api/tensorboard/{tb_id}/restart", dependencies=[Depends(auth)])
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
    gpu: str | None = None


@app.post("/api/viewer", dependencies=[Depends(auth)])
async def start_viewer(body: ViewerStart) -> dict[str, Any]:
    return await _ensure_viewer(body.latents, body.reload, body.title or "Conformational landscape",
                                reduced=body.reduced, gpu=body.gpu)


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
async def start_viewer_from_job(job_id: str, space: int | None = None,
                                gpu: str | None = None) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    program = job.get("program", "")
    args = job.get("args", {})
    # FlexConsensus: pick one consensus output space; its input space supplies the
    # full latents to decode, and the decoder network is auto-detected.
    if program == "flexconsensus":
        # Off the loop: resolving spaces can load and compare latent arrays.
        spaces = await asyncio.to_thread(_flexconsensus_spaces, job)
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
        metadata = await asyncio.to_thread(
            _find_latent_metadata, project_dir, sp["input_space"], job.get("project_id"))
        return await _ensure_viewer(sp["input_space"], sp["reload"], title, metadata,
                                    str(project_dir) if project_dir else None, sp["consensus"], gpu)
    reload_raw = args.get("pickled_nn") or args.get("reload")
    # Full latents: a direct input (annotate_space), or derived from this run's
    # own output (a predict run's predicted latents).
    latents = args.get("z_space") or args.get("latents") or args.get("latents_file")
    if not latents:
        try:
            # Off the loop: extracting the latents column parses per-particle metadata.
            latents = await asyncio.to_thread(
                adapters.produce, "latents", job,
                settings.data_dir / "viewer_cache" / job_id, "z_space")
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
    metadata = await asyncio.to_thread(
        _find_latent_metadata, project_dir, latents, job.get("project_id"))
    return await _ensure_viewer(latents, reload, title, metadata,
                                str(project_dir) if project_dir else None, reduced, gpu)


def _find_latent_metadata(project_dir: "Path | None", latents: str,
                          project_id: str | None = None) -> str | None:
    """Find the project's particle metadata that is row-aligned with these latents.

    The latents `.npy` was materialised from a predict run's `_latent_space` column;
    we recover that metadata (poses/CTF/latent) so layers can be exported with full
    per-particle info. Matched by `_latent_space` presence + row count + first-row value.

    The metadata almost always lives in a predict run's own folder, so those are
    scanned first (from the job records — a handful of directories) before falling
    back to a lazy walk of the whole project. The old version `sorted(rglob("*"))`-ed
    the entire project tree up front and parsed every metadata file on the way — a
    mature project holds hundreds of GB across millions of files, so opening the
    viewer hung the server for minutes. Blocking: call from a worker thread.
    """
    if project_dir is None:
        return None
    try:
        import numpy as np
        from xmipp_metadata.metadata import XmippMetaData

        lp = Path(latents)
        lat = np.loadtxt(lp) if lp.suffix == ".txt" else np.load(lp)
        lat = np.atleast_2d(np.asarray(lat, dtype=np.float32))

        seen: set[Path] = set()

        def _candidates():
            # Pass 1: predict-run folders, newest jobs first (deterministic).
            if project_id:
                for j in manager.list_for_project(project_id):
                    wd = j.get("workdir")
                    if not wd or j.get("program") not in _predict_programs():
                        continue
                    hits: list[Path] = []
                    for pat in ("*.xmd", "*.star"):
                        hits.extend(Path(wd).rglob(pat))
                    for p in sorted(hits):
                        if p not in seen:
                            seen.add(p)
                            yield p
            # Pass 2 (fallback): the rest of the project, lazily — no up-front
            # materialisation, and only suffix matches are ever touched.
            for pat in ("*.xmd", "*.star"):
                for p in project_dir.rglob(pat):
                    if p not in seen:
                        seen.add(p)
                        yield p

        for p in _candidates():
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


def _viewer_gpu(explicit: str | None) -> str | None:
    """The device to pin a viewer to: the caller's choice, else the saved default.

    None (nothing chosen, no default saved) means "don't pin": the viewer inherits
    our environment and JAX picks its own device, which is what it always did.
    """
    gpu = explicit if explicit is not None else settings_store.load().get("viewer_gpu")
    return (gpu or "").strip() or None


async def _ensure_viewer(latents: str, reload: str, title: str,
                         metadata: str | None = None, project_dir: str | None = None,
                         reduced: str | None = None, gpu: str | None = None) -> dict[str, Any]:
    try:
        return await viewer_manager.ensure(latents, reload, title, metadata, project_dir,
                                           reduced, _viewer_gpu(gpu))
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
_MAP_SUFFIXES = (".mrc", ".map", ".vol", ".ccp4")
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


def _find_latents_file(root: Path) -> str | None:
    """A .xmd/.star under ``root`` carrying a `_latent_space` column (name-hint first)."""
    mds = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".xmd", ".star")]
    mds.sort(key=lambda p: ("latent" not in p.name.lower(), str(p)))
    for cand in mds:
        try:
            head = cand.read_text(errors="ignore")[:20000]
        except OSError:
            continue
        if "_latent_space" in head:
            return str(cand)
    return None


def _find_sibling_latents(job: dict[str, Any]) -> tuple[str | None, str | None]:
    """Look for latents in other runs of the same project (same program first, then
    any), newest first — so attaching a *train* run (metrics+volumes) can still pull
    the landscape from the matching *predict* run. Returns (path, source_label)."""
    pid = job.get("project_id")
    if not pid:
        return None, None
    try:
        siblings = [j for j in manager.list_for_project(pid) if j.get("id") != job.get("id")]
    except Exception:  # noqa: BLE001
        return None, None
    prog = job.get("program")
    siblings.sort(key=lambda j: j.get("program") != prog)  # same-program first (stable: keeps recency)
    for sib in siblings:
        wd = sib.get("workdir")
        if not wd or not Path(wd).is_dir():
            continue
        found = _find_latents_file(Path(wd))
        if found:
            return found, (sib.get("name") or sib.get("program") or "sibling run")
    return None, None


def _resolve_run_artifacts(job: dict[str, Any]) -> dict[str, Any]:
    """Resolve a job's run folder into the artifact paths the assistant reads:
    the run dir (metrics + config), a latents metadata file, and a decoded-volume
    series directory. Best-effort — any of latents/volumes may be None. When the
    attached run has no latents of its own, they are pulled from a sibling run in
    the same project (e.g. train run ← its predict run)."""
    wd = job.get("workdir")
    run_dir = wd if wd and Path(wd).is_dir() else None
    latents: str | None = None
    latents_from: str | None = None
    volumes: str | None = None
    has_metrics = False
    if run_dir:
        root = Path(run_dir)
        # metrics = a *_metrics dir, or event files somewhere under the run folder
        has_metrics = bool(next(root.glob("*_metrics"), None)) or \
            next(root.rglob("events.out.tfevents*"), None) is not None
        latents = _find_latents_file(root)
        # volumes: the directory holding the most .mrc (≥2), preferring volume-ish names
        counts: dict[Path, int] = {}
        for p in root.rglob("*.mrc"):
            counts[p.parent] = counts.get(p.parent, 0) + 1
        series = [(d, n) for d, n in counts.items() if n >= 2]
        if series:
            def _score(item: tuple[Path, int]) -> tuple[bool, bool, int]:
                d, n = item
                nm = d.name.lower()
                # A decoded-state series with an exported deformation field wins:
                # it is a real conformational trajectory with exact correspondences,
                # unlike Intermediate_volumes (training-epoch snapshots).
                has_deform = (d / "deformation" / "deformed_points.npy").is_file()
                return (has_deform, "volume" in nm or "intermediate" in nm, n)
            volumes = str(max(series, key=_score)[0])
    if latents is None:
        latents, latents_from = _find_sibling_latents(job)
    return {
        "run_dir": run_dir,
        "latents": latents,
        "latents_from": latents_from,  # sibling run label, when latents came from one
        "volumes": volumes,
        "has_metrics": has_metrics,
        "label": f"{job.get('name') or job.get('program') or 'run'} run",
    }


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


@app.get("/api/jobs/{job_id}/usage", dependencies=[Depends(auth)])
async def job_usage(job_id: str, refresh: bool = False) -> dict[str, Any]:
    """Disk occupied by this run's folder (outputs, checkpoints and logs).

    Measured on demand rather than carried on every job listing: it means walking the
    folder, and a job list refreshes every few seconds.
    """
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    measured = await asyncio.to_thread(usage.measure, job.get("workdir"), refresh)
    return measured or {"path": None, "bytes": 0, "files": 0, "exists": False, "measured_at": None}


@app.get("/api/jobs/{job_id}/inputs", dependencies=[Depends(auth)])
def job_inputs(job_id: str) -> dict[str, Any]:
    """Volume + metadata files a job took as *inputs* (referenced in its args), so
    they can be inspected in the same viewers as the outputs."""
    return {
        "maps": _list_job_input_files(job_id, _MAP_SUFFIXES)["files"],
        "metadata": _list_job_input_files(job_id, _MD_SUFFIXES)["files"],
    }


async def _ensure_mrc_viewer(mrc: str | list[str], title: str) -> dict[str, Any]:
    try:
        return await mrc_viewer_manager.ensure(mrc, title)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"map viewer failed to start: {exc}")


class MrcViewerStart(BaseModel):
    # One map, or several to show together in one scene. `path` is kept so existing
    # single-map callers are unchanged; exactly one of the two must be given.
    path: str | None = None
    paths: list[str] | None = None
    title: str | None = None


@app.post("/api/mrcviewer", dependencies=[Depends(auth)])
async def start_mrc_viewer(body: MrcViewerStart) -> dict[str, Any]:
    """Open the map viewer for .mrc files within the sandboxed file roots.

    Several maps open as one viewer (a model panel you can toggle and morph
    through), not as several tabs."""
    wanted = body.paths if body.paths else ([body.path] if body.path else [])
    if not wanted:
        raise HTTPException(status_code=422, detail="give `path` or `paths`")
    resolved = []
    for one in wanted:
        try:
            path = filebrowser.resolve(one)   # confines to HAX_GUI_FILE_ROOTS
        except filebrowser.AccessDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"file not found: {one}")
        resolved.append(str(path))
    default = (Path(resolved[0]).name if len(resolved) == 1
               else f"{len(resolved)} maps")
    return await _ensure_mrc_viewer(resolved, body.title or default)


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
    # How many steps may run at once on any one device (GPU index, or the CPU).
    # 1 = one training per GPU; raise it only when you know two fit on one card.
    max_per_device: int | None = None


class WorkflowRun(BaseModel):
    # Optional subset of node ids to run (a single node or a selection); when
    # omitted the whole workflow runs.
    nodes: list[str] | None = None
    # Running a subset re-makes those steps' outputs, invalidating the steps they feed
    # (see cascade.py). 'queue' runs those too, after their inputs land; 'stale' just
    # flags them. Ignored when the whole graph runs — nothing is left behind to flag.
    cascade: str = "stale"
    cancel_running: bool = False
    # The user's explicit yes to stopping steps in this request that are still executing,
    # so they can be relaunched with the settings as they now stand. Without it a busy
    # step answers 409 rather than quietly letting its current run continue.
    restart_running: bool = False


class WorkflowFromProject(BaseModel):
    name: str | None = None


class WorkflowCopy(BaseModel):
    project_id: str
    name: str | None = None


@app.get("/api/workflows", dependencies=[Depends(auth)])
def list_workflows(project_id: str | None = None) -> list[dict[str, Any]]:
    return wf_manager.list(project_id)


@app.post("/api/workflows", dependencies=[Depends(auth)])
def create_workflow(body: WorkflowBody) -> dict[str, Any]:
    return wf_manager.save(body.project_id, body.name, body.graph)


@app.get("/api/workflows/resumable", dependencies=[Depends(auth)])
def resumable_workflows() -> list[dict[str, Any]]:
    """Runs the server was restarted out from under, whose next step never started.

    Declared before /api/workflows/{wid} so it isn't swallowed by it. Only workflows
    where resuming would actually *launch* something land here: one whose step is still
    running was resumed silently at startup, and one with nothing left to do was settled.
    """
    return wf_manager.list_resumable()


@app.post("/api/workflows/{wid}/resume", dependencies=[Depends(auth)])
async def resume_workflow(wid: str) -> dict[str, Any]:
    """Carry on from where the run stopped. Completed steps keep their jobs and are not
    recomputed; the steps still queued are launched in order."""
    try:
        return await wf_manager.resume(wid)
    except KeyError:
        raise HTTPException(status_code=404, detail="workflow not found")


@app.post("/api/workflows/{wid}/dismiss-resume", dependencies=[Depends(auth)])
def dismiss_workflow_resume(wid: str) -> dict[str, Any]:
    """Stop offering to resume this run (its queue is kept, so the canvas can still run it)."""
    wf = wf_manager.dismiss_resume(wid)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    return wf


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
        return wf_manager.update(wid, name=body.name, graph=body.graph,
                                 max_per_device=body.max_per_device)
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
    """Queue steps. When a run is already in flight the steps are appended to its
    queue and the live scheduler picks them up in turn.

    Running a *subset* re-makes those steps under everything they feed, so the steps
    downstream are invalidated the same way a rerun from the Jobs view invalidates them
    — and, with ``cascade='queue'``, simply join the same queue, which is where the
    dependency ordering they need already lives."""
    try:
        nodes = body.nodes
        if nodes:
            # From the graph, not the jobs' bindings: a step wired to a not-yet-run one
            # has no binding to find yet, and it is still downstream.
            down = wf_manager.downstream_nodes(wid, nodes)
            if down:
                deps = [j for j in (manager.get(jid) for jid in wf_manager.jobs_for_nodes(wid, down)) if j]
                await cascade.invalidate(deps, cancel_running=body.cancel_running)
                if body.cascade == "queue":
                    nodes = list(nodes) + down
        return await wf_manager.run(wid, nodes, restart_running=body.restart_running)
    except StepsRunning as exc:
        raise HTTPException(status_code=409, detail={
            "error": "those steps are still running", "steps": exc.steps})
    except cascade.DependentsRunning as exc:
        raise HTTPException(status_code=409, detail={
            "error": "downstream runs are still executing", "running": exc.running})
    except KeyError:
        raise HTTPException(status_code=404, detail="workflow not found")
    except WorkflowError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})


@app.post("/api/workflows/{wid}/stop", dependencies=[Depends(auth)])
async def stop_workflow(wid: str) -> dict[str, Any]:
    """Halt a run: cancel the step executing now and dequeue everything still waiting."""
    try:
        return await wf_manager.stop(wid)
    except KeyError:
        raise HTTPException(status_code=404, detail="workflow not found")


@app.post("/api/workflows/from-project/{project_id}", dependencies=[Depends(auth)])
def workflow_from_project(project_id: str, body: WorkflowFromProject = WorkflowFromProject()) -> dict[str, Any]:
    """Open a project as a *live* workflow (get-or-create): nodes = the project's
    jobs, edges inferred from input bindings, kept in sync with the project. Editing
    / running / labeling this workflow modifies the actual project."""
    proj = _project_or_404(project_id)
    return wf_manager.open_project(project_id, body.name or proj["name"])


@app.post("/api/workflows/{wid}/copy", dependencies=[Depends(auth)])
def workflow_copy(wid: str, body: WorkflowCopy) -> dict[str, Any]:
    """Clone a workflow's graph into a new standalone workflow owned by another
    project. Self-contained: node/job links dropped, external data inputs cleared,
    internal dependencies preserved so it re-runs cleanly in the target project."""
    if wf_manager.get(wid) is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    _project_or_404(body.project_id)
    try:
        return wf_manager.copy_to_project(wid, body.project_id, body.name)
    except KeyError:
        raise HTTPException(status_code=404, detail="target project not found")


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
    # The canvas plotter and the latent scatter live in the viewer app's static dir.
    # The SPA's metrics view imports the very same modules from /shared, rather than
    # keeping a forked copy in sync by hand.
    shared = Path(__file__).resolve().parent / "viewer_app" / "static"
    if shared.is_dir():
        app.mount("/shared", StaticFiles(directory=str(shared)), name="shared")

    frontend = settings.frontend_dir
    if (frontend / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="spa")


mount_spa()
