"""Optional bridge to the ``hax-em-assistant`` package.

The GUI must run whether or not the assistant is installed, so everything here is
guarded: :func:`available` reports installation, :func:`status` returns a plain
dict (``{"available": False}`` when absent), and the chat streamer is only reached
after an availability check. Nothing at import time requires the assistant.
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from . import settings_store
from .config import settings

try:  # the assistant is an independent, optional package
    from hax_assistant.session import AssistantSession
    _INSTALLED = True
except Exception:  # noqa: BLE001 - ImportError, or a broken/incompatible install
    AssistantSession = None  # type: ignore[assignment,misc]
    _INSTALLED = False

_session: Optional["AssistantSession"] = None
_session_lock = threading.Lock()
_schema_index: Any = None          # False once loading it failed
_facts_cache: dict[tuple, dict] = {}


class Refused(Exception):
    """The assistant's safety rules refuse an action; the message is meant for the user."""


def available() -> bool:
    return _INSTALLED


def _get() -> "AssistantSession":
    global _session
    # Chats stream in parallel: build the one session (and its lazily cached tier) once.
    with _session_lock:
        if _session is None:
            session = AssistantSession()  # type: ignore[operator]
            session.tier
            _session = session
    return _session


def base_url() -> str:
    """The configured Ollama endpoint, read from config alone.

    Deliberately does NOT construct the session: construction builds the docs
    vector index, whose embeddings need Ollama up — the caller uses this to
    start Ollama *first* (see the warmup task in app.py). ``status()`` here
    would re-create the old ordering bug it exists to fix.
    """
    if not _INSTALLED:
        return ""
    try:
        from hax_assistant.config import Settings
        return Settings.from_env().base_url
    except Exception:  # noqa: BLE001
        return ""


def status() -> dict:
    """Availability + (when installed) model/grounding info for the SPA."""
    if not _INSTALLED:
        return {"available": False}
    try:
        st = {"available": True, **_get().status()}
    except Exception as e:  # noqa: BLE001 - surface a bad config without crashing the GUI
        return {"available": False, "error": str(e)}
    st["workspace_error"] = ensure_workspace()
    return st


def ensure_workspace() -> Optional[str]:
    """Register the GUI's projects folder as an assistant workspace. Returns why the
    assistant may not act on runs there (None when it may)."""
    if not _INSTALLED:
        return "the assistant is not installed"
    root = settings_store.projects_root()
    if root is None:
        return "no projects folder is set in Settings, so the assistant may not act on any run"
    try:
        from hax_assistant import safety
        safety.add_workspace(root)
    except Exception as e:  # noqa: BLE001 - SafetyError (e.g. too broad) or OSError
        return str(e)
    return None


def get_autonomy() -> dict:
    from hax_assistant import autonomy
    level = autonomy.get_autonomy()
    return {"level": level, "description": autonomy.DESCRIPTIONS[level],
            "levels": {k: autonomy.DESCRIPTIONS[k] for k in autonomy.LEVELS},
            "env_override": bool(os.environ.get("HAX_ASSISTANT_AUTONOMY"))}


def set_autonomy(level: str) -> dict:
    """Raises ValueError on an unknown level."""
    from hax_assistant import autonomy
    autonomy.set_autonomy(level)
    return get_autonomy()


def data_facts(path: str) -> dict:
    """The assistant's deterministic facts about a data path ({} when unavailable)."""
    if not _INSTALLED:
        return {}
    try:
        st = Path(path).expanduser().stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        facts = _facts_cache.get(key)        # a local: another turn may clear the cache meanwhile
        if facts is None:
            from hax_assistant.context.data import inspect_data
            if len(_facts_cache) > 64:
                _facts_cache.clear()
            facts = _facts_cache[key] = inspect_data(str(path)).as_dict()
        return facts
    except Exception:  # noqa: BLE001 - facts are optional
        return {}


# -- run monitoring (deterministic rules + remedies, no LLM) -----------------
def _schema():
    """The assistant's schema index if cheaply available (None otherwise)."""
    global _schema_index
    if _session is not None and _session.schema is not None:
        return _session.schema
    if _schema_index is None:
        try:
            from hax_assistant.config import Settings, resolve_schema_path
            from hax_assistant.knowledge.schema import load_schema_index
            path, source = resolve_schema_path(Settings.from_env())
            _schema_index = load_schema_index(path, source=source)
        except Exception:  # noqa: BLE001
            _schema_index = False
    return _schema_index or None


def _finding(f) -> dict:
    return {"kind": f.kind, "severity": f.severity, "title": f.title,
            "evidence": f.evidence, "calibrated": f.calibrated}


def _proposal(finding, remedy) -> dict:
    from hax_assistant.monitor import explain
    return {"kind": finding.kind, "finding": _finding(finding), "restart": remedy.restart,
            "changes": dict(remedy.changes), "reason": remedy.reason,
            "details": list(remedy.details), "text": explain(finding, remedy)}


def watch_report(run_dir: str, pid_alive: Optional[bool] = True) -> dict:
    """Health findings for a run and the restart the assistant would propose, if any."""
    from hax_assistant.monitor import check_health, pick, read_signals
    signals = read_signals(run_dir)
    findings = check_health(signals, pid_alive=pid_alive)
    choice = pick(findings, signals, _schema())
    return {"findings": [_finding(f) for f in findings],
            "proposal": _proposal(*choice) if choice and choice[1].restart != "none" else None}


def remedy_for(run_dir: str, kind: str, pid_alive: Optional[bool] = None) -> Optional[dict]:
    """Re-derive, from the run's own signals, the restart for finding ``kind`` (None if none)."""
    from hax_assistant.monitor import check_health, propose_remedy, read_signals
    signals = read_signals(run_dir)
    finding = next((f for f in check_health(signals, pid_alive=pid_alive) if f.kind == kind), None)
    remedy = propose_remedy(finding, signals, _schema()) if finding is not None else None
    if remedy is None or remedy.restart == "none":
        return None
    return _proposal(finding, remedy)


def decide_relaunch(run_dir: str) -> str:
    """``propose`` | ``confirm`` | ``proceed`` for relaunching ``run_dir`` at the current autonomy."""
    from hax_assistant import safety
    ensure_workspace()
    return safety.decide("relaunch", run_dir)


def decide_stop(run_dir: str, level: Optional[str] = None) -> str:
    """``propose`` | ``confirm`` | ``proceed`` for stopping the job in ``run_dir`` at ``level`` (default:
    the current autonomy)."""
    from hax_assistant import safety
    ensure_workspace()
    return safety.decide("stop", run_dir, level)


def guard_relaunch(program: str, run_dir: str, changes: dict) -> None:
    """Raise :class:`Refused` unless the assistant may relaunch this run with ``changes``."""
    from hax_assistant import safety
    ensure_workspace()
    try:
        safety.require_program(program, _schema())
        safety.require_inside(run_dir, "run folder")
        safety.require_free_changes(program, changes, _schema())
    except safety.SafetyError as e:
        raise Refused(str(e)) from None


def relaunch_budget(origin_dir: str) -> tuple[int, int]:
    """``(used, allowed)`` relaunches for the run a chain of relaunches started from."""
    from hax_assistant import autonomy
    from hax_assistant.monitor.controller import count_relaunches
    allowed = int(autonomy.get_budget("max_relaunches", autonomy.MAX_RELAUNCHES_DEFAULT))
    return count_relaunches(origin_dir), allowed


def applied_before(origin_dir: str, job_id: str, kind: str) -> bool:
    """Whether the audit trail already records relaunching ``job_id`` for finding ``kind``."""
    from hax_assistant.monitor.controller import AUDIT_FILE
    try:
        lines = (Path(origin_dir) / AUDIT_FILE).read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if (isinstance(rec, dict) and rec.get("decision") == "relaunched" and rec.get("job_id") == job_id
                and (rec.get("finding") or {}).get("kind") == kind):
            return True
    return False


def autonomy_level() -> str:
    from hax_assistant import autonomy
    return autonomy.get_autonomy()


def decide_launch() -> str:
    """``propose`` | ``confirm`` | ``proceed`` for launching a workflow at the current autonomy."""
    from hax_assistant import autonomy
    return autonomy.decide("launch")


def plan_workflow(request: str, project_dir: str, name: Optional[str] = None,
                  gpu: Optional[str] = None) -> dict:
    """Re-plan a workflow request for a GUI project folder, as plain data. Deterministic and
    creates nothing; raises :class:`Refused` when the request is not a workflow it can plan.
    With an SSD scratch root in Settings the steps read their images from the project's scratch
    folder (the job manager fills --ssd_scratch_folder), not from RAM."""
    from hax_assistant.workflows import Check, WorkflowError, plan_request
    ensure_workspace()
    root = settings_store.ssd_scratch_root()
    try:
        plan = plan_request(request, _schema(), project_dir=project_dir, name=name, gpu=gpu,
                            cli=shlex.split(settings.hax_cli),
                            scratch=str(root / Path(project_dir).name) if root else None,
                            scratch_name="the GUI's SSD scratch folder, set in Settings")
    except WorkflowError as e:
        raise Refused(str(e)) from None
    if root is None:
        plan.checks.append(Check("SSD scratch", "warn", "no SSD scratch folder is set in Settings: set one and "
                                                        "the jobs read their images from that fast disk instead "
                                                        "of loading them into RAM or copying them into each run"))
    steps = [{"index": s.index, "key": s.key, "program": s.program, "mode": s.mode, "title": s.title,
              "internal": s.internal, "args": dict(s.args), "params": dict(s.params),
              "reasons": dict(s.reasons), "links": dict(s.links), "watch": s.watch, "note": s.note}
             for s in plan.steps]
    return {"template": plan.template, "description": plan.description, "name": plan.name,
            "gpu": plan.gpu, "effort": plan.effort, "inputs": plan.inputs, "reasons": plan.reasons,
            "steps": steps, "checks": [asdict(c) for c in plan.checks], "ok": plan.ok,
            "describe": plan.describe(commands=False)}


def audit(origin_dir: str, record: dict) -> None:
    from hax_assistant.monitor.controller import append_audit
    append_audit(origin_dir, record)


def register_job(output_dir: str, pid: int, argv: list[str], origin: Optional[str] = None) -> None:
    from hax_assistant import safety
    safety.register_job(output_dir, pid, argv, origin=origin)


def reindex_docs(*, force: bool = True) -> dict:
    """Rebuild the docs vector index in place from the current hax-em-docs
    checkout, so doc edits are picked up without restarting the GUI process."""
    if not _INSTALLED:
        return {"ok": False, "reason": "assistant not installed", "chunks": 0}
    try:
        return _get().reindex_docs(force=force)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": str(e), "chunks": 0}


def diagnose_job_log(log_text: str, run_config: str | None = None) -> dict:
    """Diagnose a failed run's log: deterministic classifier findings always,
    an LLM root-cause/fix narrative when the model is reachable.

    The classifier (hax_assistant.context.logs) is a rule table — GPU OOM, shape
    mismatch, NaN divergence, missing metadata labels, checkpoint errors — so
    the findings arrive even with Ollama down; the narrative is best-effort.
    """
    if not _INSTALLED:
        return {"available": False}
    from hax_assistant.context.logs import classify_log
    findings = [{"kind": f.kind, "title": f.title, "hint": f.hint, "evidence": f.evidence}
                for f in classify_log(log_text)]
    analysis, error = None, None
    try:
        from hax_assistant.skills.diagnose_logs import diagnose_logs
        analysis = diagnose_logs(_get().backend, log_text, run_config)
    except Exception as e:  # noqa: BLE001 - the findings alone are still useful
        error = str(e)
    return {"available": True, "findings": findings, "analysis": analysis,
            "analysis_error": error}


def list_gpus() -> list[dict]:
    """Visible GPUs (for the prepare-motion confirmation's GPU picker)."""
    if not _INSTALLED:
        return []
    try:
        from hax_assistant.actions import list_gpus as _lg
        return _lg()
    except Exception:  # noqa: BLE001
        return []


def plan_prepare_motion(run_dir: str) -> dict:
    """Read-only plan of what preparing regional motion for a run would do."""
    from hax_assistant.actions import plan_motion_preparation
    return plan_motion_preparation(run_dir)


def run_prepare_motion(run_dir: str, gpu, *, on_progress=None) -> str:
    """Execute the (confirmed) prepare-motion pipeline; returns the deformation dir.
    Raises :class:`Refused` when ``run_dir`` is outside the assistant's workspaces."""
    from hax_assistant import safety
    from hax_assistant.actions import prepare_motion
    reason = ensure_workspace()
    try:
        safety.require_inside(run_dir, "run folder")
    except safety.SafetyError as e:
        raise Refused(str(e) + (f" ({reason})" if reason else "")) from None
    return prepare_motion(run_dir, gpu, on_progress=on_progress)


def stream_reply(messages: list[dict], grounded: bool = True,
                 run: Optional[dict] = None) -> Iterator[str]:
    """Yield reply tokens for a conversation (list of {role, content})."""
    return _get().stream_reply(messages, grounded=grounded, run=run)


def stream_events(messages: list[dict], grounded: bool = True,
                  run: Optional[dict] = None, allow_actions: bool = False,
                  interaction=None, tools=None) -> Iterator[dict]:
    """Yield typed events (status activity + content tokens + action proposals).

    ``run`` (optional) is a dict of resolved artifact paths for an attached run:
    ``{"run_dir", "latents", "volumes", "label"}`` — the session loads those
    deterministically and injects a features summary for the model to narrate.
    ``allow_actions`` lets the session emit deterministic action proposals (create
    a draft job from a data path); the GUI confirms before anything happens.
    ``interaction`` (a :class:`ChatInteraction`) lets the session ask the user.
    ``tools`` replace the free-tier tools of the same name (see assistant_tools).
    """
    return _get().stream_events(messages, grounded=grounded, run=run, allow_actions=allow_actions,
                                interaction=interaction, tools=tools, host_notes=scratch_note())


def scratch_note() -> str:
    """What the model must know about the GUI's SSD scratch folder when it suggests flags."""
    if settings_store.ssd_scratch_root() is not None:
        return ("This GUI has an SSD scratch folder (set in its Settings): every job it launches gets "
                "--ssd_scratch_folder automatically, one folder per project. Never ask the user for a scratch "
                "path and never set --ssd_scratch_folder yourself; leave --load_images_to_ram off, so the "
                "images are read from that folder.")
    return ("No SSD scratch folder is set in this GUI's Settings, so images are loaded into RAM "
            "(--load_images_to_ram) or copied into each run folder. Suggest setting an SSD scratch folder in "
            "Settings rather than asking the user for a path.")


class ChatInteraction:
    """Ask-and-wait over the chat socket: :meth:`ask` (called from the worker thread)
    sends a question and blocks until the client answers, the turn's ``cancel`` event
    is set (a stop, a new question, or a disconnect), or the attempts run out."""

    max_attempts = 3

    def __init__(self, send: Callable[[str, Any], None], cancel: threading.Event) -> None:
        self._send = send              # thread-safe (kind, payload) -> None
        self._cancel = cancel
        self._lock = threading.Lock()
        self._pending: Optional[tuple[str, threading.Event, dict]] = None

    def ask(self, q) -> str:
        from hax_assistant.interaction import Cancelled, validate_answer
        error = None
        for _ in range(self.max_attempts):
            qid, got, box = uuid.uuid4().hex[:12], threading.Event(), {}
            with self._lock:
                self._pending = (qid, got, box)
            self._send("question", {"id": qid, "key": q.key, "text": q.text, "kind": q.kind,
                                    "choices": list(q.choices), "default": q.default,
                                    "error": error})
            try:
                while not got.wait(0.1):
                    if self._cancel.is_set():
                        raise Cancelled()
            finally:
                with self._lock:
                    self._pending = None
            if box.get("cancel") or self._cancel.is_set():
                raise Cancelled()
            try:
                return validate_answer(q, "" if box.get("value") is None else str(box["value"]))
            except ValueError as e:
                error = str(e)
        raise Cancelled()

    def answer(self, data: Any) -> bool:
        """Route a client ``{"id", "value"}`` (or ``{"id", "cancel": true}``) to the
        pending question. False when it answers nothing (stale or malformed)."""
        if not isinstance(data, dict):
            return False
        with self._lock:
            if self._pending is None or data.get("id") != self._pending[0]:
                return False
            _, got, box = self._pending
            box.update(value=data.get("value"), cancel=bool(data.get("cancel")))
        got.set()
        return True

    def release(self) -> None:
        """Wake a pending :meth:`ask` at once after the turn's cancel is set (it raises Cancelled)."""
        with self._lock:
            pending = self._pending
        if pending is not None:
            pending[1].set()

    def notify(self, text: str, level: str = "info") -> None:
        self._send("status", text)


def generate_report(out_path: str, *, run_dir: Optional[str] = None,
                    latents: Optional[str] = None, volumes: Optional[str] = None,
                    title: str = "hax-em run report", subtitle: str = "",
                    narrate: bool = True) -> dict:
    """Build a deterministic run-report PDF at ``out_path`` (figures + measured
    values, per-domain and per-particle motion, 3D views), optionally with an
    AI-generated discussion section, through the assistant's shared collector.
    The 3D views, ChimeraX files and per-particle table go to ``<stem>_files/``
    next to it. Returns the report manifest. Requires the assistant package."""
    from hax_assistant.report.collect import collect_run, write_analysis

    out = Path(out_path)
    return write_analysis(collect_run(run_dir, latents=latents, volumes=volumes),
                          out.with_name(out.stem + "_files"), _get(), narrate=narrate,
                          pdf_path=out, title=title, subtitle=subtitle)
