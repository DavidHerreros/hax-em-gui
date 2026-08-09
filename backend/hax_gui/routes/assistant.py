"""Assistant routes: status, docs reindex, action previews, prepare-motion,
run reports, the chat WebSocket, and chat history CRUD."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import assistant as assistant_bridge
from .. import assistant_actions
from .. import gpus as gpus_mod
from .. import chats as chat_store
from .. import ollama as ollama_manager_mod
from ..config import settings
from ..jobs import manager
from .artifacts import _resolve_run_artifacts
from .shared import auth, ws_authorized as _ws_authorized

router = APIRouter()

# -- optional AI assistant -------------------------------------------------
@router.get("/api/assistant/status", dependencies=[Depends(auth)])
def assistant_status() -> dict[str, Any]:
    """Availability + config of the optional hax-em-assistant. Always present so
    the SPA can decide whether to show the chat panel; reports available=False
    (never errors) when the package is not installed."""
    st = assistant_bridge.status()
    st["actions_enabled"] = assistant_actions.actions_enabled()
    return st


@router.post("/api/assistant/reindex-docs", dependencies=[Depends(auth)])
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


@router.post("/api/assistant/actions/preview-job", dependencies=[Depends(auth)])
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


@router.get("/api/assistant/actions/prepare-motion/{job_id}", dependencies=[Depends(auth)])
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


@router.websocket("/ws/assistant/prepare-motion")
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


@router.get("/api/assistant/run-preview/{job_id}", dependencies=[Depends(auth)])
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


@router.post("/api/assistant/run-report/{job_id}", dependencies=[Depends(auth)])
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


@router.get("/api/ollama/status", dependencies=[Depends(auth)])
def ollama_status(request: Request) -> dict[str, Any]:
    """Whether a local Ollama is running and whether the GUI is managing it."""
    base_url = ""
    if assistant_bridge.available():
        try:
            base_url = assistant_bridge.status().get("base_url", "")
        except Exception:  # noqa: BLE001
            base_url = ""
    out: dict[str, Any] = {"manage_enabled": settings.manage_ollama,
                           "startup": getattr(request.app.state, "ollama", {})}
    if base_url:
        out.update(ollama_manager_mod.manager.status(base_url))
    return out


@router.websocket("/ws/assistant/chat")
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
            # Client-generated turn id, echoed on the final done so the client
            # can tell a cancelled turn's done from the live turn's (a stop
            # followed immediately by a new question overlaps the two).
            turn_id = req.get("turn")
            # Optional attached run: resolve the job id to artifact paths server-side
            # (the client never handles filesystem paths).
            run = None
            run_job_id = req.get("run_job_id")
            if run_job_id:
                job = manager.get(run_job_id)
                if job is not None:
                    run = _resolve_run_artifacts(job)
                    run["job_id"] = run_job_id   # so a proposed action can target it
            # Extra attached runs (cross-run reasoning): they contribute measured
            # facts + a parameter diff against the primary, not full artifacts —
            # the primary stays the run whose landscape/metrics the session loads.
            extra_jobs = [j for j in (manager.get(i) for i in (req.get("extra_run_job_ids") or [])[:2])
                          if j is not None and j["id"] != run_job_id]
            allow_actions = assistant_actions.actions_enabled()

            # Run the blocking LLM stream off the event loop, funnel events back.
            # A cancel Event lets a client "stop" actually halt generation (the
            # worker closes the generator, which tears down the model request).
            out: asyncio.Queue = asyncio.Queue()
            cancel = threading.Event()

            def _worker(messages=messages, grounded=grounded, run=run, job=job if run_job_id else None,
                        extra_jobs=extra_jobs, allow_actions=allow_actions, cancel=cancel, out=out) -> None:
                # EVERYTHING inside the try, including building the generator:
                # session construction (docs index load, embedder init) can
                # raise, and with it outside the try neither ("error") nor the
                # ("done") sentinel was ever queued — the socket then hung
                # forever on out.get(), with the client stuck on "Stop".
                gen = None
                try:
                    if run is not None and job is not None:
                        # Off the loop (metadata heads + nvidia-smi): measured data
                        # facts + free VRAM the model can ground advice in.
                        try:
                            run["facts"] = _run_facts(job)
                        except Exception:  # noqa: BLE001 - facts are optional
                            pass
                        # Cross-run grounding: each extra run adds its own measured
                        # facts and a deterministic parameter diff vs the primary,
                        # so "why did run B beat run A" is answered from data.
                        for ej in extra_jobs:
                            try:
                                run["facts"] = (run.get("facts") or "") + "\n\n" + _extra_run_facts(job, ej)
                            except Exception:  # noqa: BLE001
                                pass
                    gen = assistant_bridge.stream_events(messages, grounded=grounded, run=run,
                                                         allow_actions=allow_actions)
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
                    if gen is not None:
                        try:
                            gen.close()  # stop the model stream promptly on cancel
                        except Exception:  # noqa: BLE001 - the sentinel must go out
                            pass
                    loop.call_soon_threadsafe(out.put_nowait, ("done", None))

            loop.run_in_executor(None, _worker)

            # Forward events while watching the inbox. An explicit {"stop":…}
            # cancels this turn; a NEW turn ({"messages":…} — the user stopped
            # and immediately asked again) cancels this one and is carried over
            # to run next. The old behaviour treated *any* inbound frame as a
            # stop and threw its payload away, so a question sent right after
            # Stop vanished permanently with no error. Keep draining after a
            # cancel until the worker reports done, but stop forwarding tokens.
            get_task = asyncio.ensure_future(out.get())
            inbox_task = asyncio.ensure_future(inbox.get())
            stopped = False
            disconnected = False
            carry: dict | None = None
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
                            if isinstance(msg, dict) and "messages" in msg:
                                carry = msg        # latest next-turn request wins
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
            await websocket.send_json({"done": True, "turn": turn_id})
            if carry is not None:
                inbox.put_nowait(carry)   # run the question that arrived mid-stop
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


@router.get("/api/chats", dependencies=[Depends(auth)])
def list_chats(project_id: str | None = None) -> list[dict[str, Any]]:
    return chat_store.manager.list(project_id)


@router.post("/api/chats", dependencies=[Depends(auth)])
def create_chat(body: ChatCreate) -> dict[str, Any]:
    return chat_store.manager.create(body.project_id, body.title)


@router.get("/api/chats/{cid}", dependencies=[Depends(auth)])
def get_chat(cid: str) -> dict[str, Any]:
    chat = chat_store.manager.get(cid)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return chat


@router.put("/api/chats/{cid}", dependencies=[Depends(auth)])
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


@router.delete("/api/chats/{cid}", dependencies=[Depends(auth)])
def delete_chat(cid: str) -> dict[str, Any]:
    if not chat_store.manager.delete(cid):
        raise HTTPException(status_code=404, detail="chat not found")
    return {"ok": True}



def _run_facts(job: dict[str, Any], include_gpus: bool = True) -> str:
    """Measured facts about the attached run's data and this machine's GPUs,
    for the model to ground parameter advice in (P6.2). Deterministic and
    cheap: metadata heads and MRC headers only, plus one nvidia-smi query.
    `include_gpus=False` for extra attached runs — the GPUs are machine-global
    and repeating them per run is noise."""
    lines: list[str] = []
    args = job.get("args") or {}
    for field in ("md", "vol", "mask"):
        val = args.get(field)
        if not isinstance(val, str) or not val:
            continue
        facts = assistant_actions.inspect_path(val)
        if not facts.get("exists"):
            lines.append(f"--{field}: {val} (MISSING on disk)")
            continue
        bits = [f"--{field}: {val}"]
        for key in ("kind", "particles", "box", "voxel_size", "columns_count",
                    "has_ctf", "has_latents", "volumes", "size"):
            if key in facts:
                bits.append(f"{key}={facts[key]}")
        if "columns" in facts:
            bits.append(f"columns={','.join(facts['columns'][:12])}")
        lines.append("  ".join(bits))
    if include_gpus:
        for g in gpus_mod.list_gpus():
            used, total = g.get("memory_used_mb"), g.get("memory_total_mb")
            free = (total - used) if (used is not None and total is not None) else None
            lines.append(f"GPU {g.get('index')}: {g.get('name')}"
                         + (f", {free} MB free of {total} MB" if free is not None else ""))
    if job.get("gpu"):
        lines.append(f"This run is pinned to GPU {job['gpu']}.")
    return "\n".join(lines)


def _extra_run_facts(primary: dict[str, Any], other: dict[str, Any]) -> str:
    """Cross-run grounding block for an additionally attached run: its identity
    and outcome, the flags on which it differs from the primary run (the
    question multi-attach exists to answer), and its own measured data facts."""
    label = other.get("name") or other["program"]
    lines = [f"=== ALSO ATTACHED RUN: {label} ({other['program']}, status {other['status']}"
             + (f", exit {other['return_code']}" if other.get("return_code") is not None else "") + ") ==="]
    pa, pb = primary.get("args") or {}, other.get("args") or {}
    diffs = [f"--{k}: {pa.get(k, '(unset)')} vs {pb.get(k, '(unset)')}"
             for k in sorted(set(pa) | set(pb)) if str(pa.get(k)) != str(pb.get(k))]
    lines.append("Parameter differences vs the primary run: "
                 + ("; ".join(diffs[:25]) if diffs else "none — identical arguments."))
    pkgs = (other.get("provenance") or {}).get("packages")
    if pkgs:
        lines.append("Recorded versions: " + ", ".join(f"{k} {v}" for k, v in pkgs.items()))
    facts = _run_facts(other, include_gpus=False)
    if facts:
        lines.append(facts)
    return "\n".join(lines)


@router.post("/api/jobs/{job_id}/diagnose", dependencies=[Depends(auth)])
async def diagnose_job(job_id: str) -> dict[str, Any]:
    """Diagnose a failed run from its log (assistant's classifier + LLM).

    The classifier findings are deterministic and come back even when the model
    is unreachable; the narrative can take tens of seconds (one LLM call), so
    the whole thing runs off the event loop."""
    if not assistant_bridge.available():
        raise HTTPException(status_code=404, detail="assistant not installed")
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    def _run() -> dict[str, Any]:
        # stderr usually holds the traceback; stdout the run context. Feed both.
        err = "\n".join(manager.tail(job_id, max_lines=200, stream="err"))
        out = "\n".join(manager.tail(job_id, max_lines=200, stream="out"))
        log_text = (out + "\n" + err).strip()
        config = " ".join(job.get("command") or [])
        return assistant_bridge.diagnose_job_log(log_text, config)

    result = await asyncio.to_thread(_run)
    return {**result, "job_id": job_id, "status": job["status"]}
