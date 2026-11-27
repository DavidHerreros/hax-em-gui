"""The run monitor and the chat.

A job can have a chat: the one whose assistant launched it (a workflow card or the free-tier tools),
or the one the user asked to watch it from. The monitor's findings for that job are posted into that
chat as an assistant message, with the same Apply as the notification (one decision, wherever it is
clicked). With a model that has tool calling (the free tier) the assistant then reviews each finding
in that chat, through the same gates as a turn the user started, and nobody is asked mid-turn: a
change that needs an OK becomes an Apply in the chat. At auto the watcher leaves the decision to that
review; if the review fails it acts as before.

Links, the workflows the assistant launched and the decisions on findings are kept in
``<data_dir>/assistant_monitor.json``. Watching itself lives in memory (watch.py), so at startup the
linked jobs still running are watched again and the launched workflows followed again.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any

from . import assistant as assistant_bridge
from . import chats as chat_store
from . import events, jsonfile
from .config import settings
from .jobs import ACTIVE, manager

FILE = "assistant_monitor.json"
MAX_REVIEWS_PER_JOB = 3
USER_RUN = ("You started this run, so the assistant only proposes, at every autonomy level: nothing changes "
            "until you click Apply, and a relaunch is a new job (your run folder is kept).")
_busy: dict[str, int] = {}            # chat id -> turns the user has running there
_waiting: dict[str, list] = {}        # chat id -> reviews held until those turns end
_tasks: set[asyncio.Task] = set()


# ── the store ────────────────────────────────────────────────────────────────
def _load() -> dict[str, Any]:
    data = jsonfile.read(settings.data_dir / FILE, {})
    data = data if isinstance(data, dict) else {}
    for key in ("jobs", "workflows", "findings"):
        data.setdefault(key, {})
    return data


def _save(data: dict[str, Any]) -> None:
    jsonfile.write(settings.data_dir / FILE, data)


def link(job_id: str, chat_id: str | None, **step: Any) -> str | None:
    """Post ``job_id``'s findings to ``chat_id`` from now on (``step``: its workflow and step, when the
    assistant launched it). Returns the chat it was linked to before, when that was another one."""
    data = _load()
    rec = data["jobs"].setdefault(job_id, {})
    before = rec.get("chat_id")
    rec.update(step)
    if chat_id or "chat_id" not in rec:
        rec["chat_id"] = chat_id
    _save(data)
    return before if before and before != rec["chat_id"] else None


def forget_chat(job_id: str) -> str | None:
    data = _load()
    rec = data["jobs"].get(job_id) or {}
    before = rec.pop("chat_id", None)
    if before:
        _save(data)
    return before


def chat_of(job_id: str) -> str | None:
    return (_load()["jobs"].get(job_id) or {}).get("chat_id")


def step_of(job_id: str) -> dict[str, Any]:
    """The workflow and step of a job the assistant launched ({} otherwise)."""
    rec = _load()["jobs"].get(job_id) or {}
    return {k: rec[k] for k in ("workflow_id", "workflow", "step") if k in rec}


def jobs_of_workflow(ref: str) -> set[str]:
    """The jobs of a workflow the assistant launched, by its id or name."""
    ref = ref.lower()
    return {jid for jid, rec in _load()["jobs"].items()
            if ref in (str(rec.get("workflow_id", "")).lower(), str(rec.get("workflow", "")).lower())}


def remember_workflow(wid: str, chat_id: str | None, origin: str, steps: list[dict[str, Any]]) -> None:
    data = _load()
    data["workflows"][wid] = {"chat_id": chat_id, "origin": origin,
                              "steps": [{k: s[k] for k in ("node", "title", "watch")} for s in steps]}
    _save(data)


def forget_workflow(wid: str) -> None:
    data = _load()
    if data["workflows"].pop(wid, None) is not None:
        _save(data)


def _finding(job_id: str, kind: str) -> dict[str, Any] | None:
    return _load()["findings"].get(f"{job_id}:{kind}")


def _update_finding(job_id: str, kind: str, **fields: Any) -> dict[str, Any]:
    data = _load()
    rec = data["findings"].setdefault(f"{job_id}:{kind}", {})
    rec.update(fields)
    _save(data)
    return rec


def is_resolved(job_id: str, kind: str) -> bool:
    return bool((_finding(job_id, kind) or {}).get("resolved"))


def resolve(job_id: str, kind: str, result: dict[str, Any]) -> None:
    """The finding was decided (applied, from the chat, the notification or on its own): mark it in its
    chat messages and tell the open views, so it is never applied twice."""
    rec = _finding(job_id, kind)
    if rec is None:
        return
    rec = _update_finding(job_id, kind, resolved=result)
    for mid in rec.get("messages", []):
        chat_store.manager.patch_posted(rec["chat_id"], mid, resolved=result)
    events.publish({"kind": "watch_resolved", "job_id": job_id, "alert": kind, "message": result.get("message")})
    events.publish({"kind": "chat", "chat_id": rec["chat_id"]})


def _post(chat_id: str, job_id: str, kind: str, text: str, posted: str, **card: Any) -> str | None:
    mid = uuid.uuid4().hex[:12]
    msg = {"role": "assistant", "content": text, "posted": posted, "id": mid,
           "watch": {"job_id": job_id, "kind": kind, **card}}
    if chat_store.manager.post(chat_id, msg) is None:
        forget_chat(job_id)                # the chat is gone: back to the notification only
        return None
    rec = _finding(job_id, kind) or {}
    _update_finding(job_id, kind, chat_id=chat_id, messages=rec.get("messages", []) + [mid])
    events.publish({"kind": "chat", "chat_id": chat_id})
    return mid


# ── part 1: a finding into its chat ──────────────────────────────────────────
def _owned(job: dict[str, Any]) -> bool:
    try:
        return assistant_bridge.owned(job.get("workdir") or "")
    except Exception:  # noqa: BLE001 - unknown ownership counts as the user's
        return False


def post_finding(job: dict[str, Any], finding: dict[str, Any], event: dict[str, Any]) -> None:
    """Post a watch alert into the job's chat, if it has one, and start the assistant's review of it."""
    chat_id = chat_of(job["id"])
    if not chat_id:
        return
    proposal = event.get("proposal")
    lines = [f"**Run monitor** — {event['title']}"]
    if finding.get("evidence"):
        lines.append(str(finding["evidence"]))
    if proposal:
        lines.append(f"Proposal: {proposal['text']}")
    if event.get("acted"):
        lines.append(f"Done automatically: {event['acted']['message']}")
    elif event.get("deferred"):
        lines.append("Autonomy is auto: the assistant reviews it first and acts on its own only through the same "
                     "rules (a calibrated finding, a job it launched).")
    elif event.get("needs_confirmation"):
        lines.append("Nothing changes until you click Apply (here or in the notification).")
    elif event.get("blocked"):
        lines.append(f"Not acted on: {event['blocked']}")
    if proposal and not _owned(job):
        lines.append(USER_RUN)
    acted = event.get("acted")
    _post(chat_id, job["id"], finding["kind"], "\n\n".join(lines), "watch", name=event.get("name"),
          proposal=proposal, needs_confirmation=bool(event.get("needs_confirmation")),
          deferred=bool(event.get("deferred")), resolved=acted if acted and acted.get("applied") else None)
    if acted and acted.get("applied"):
        _update_finding(job["id"], finding["kind"], resolved=acted)
    if will_review(job["id"], finding["kind"]):
        _schedule(chat_id, job, finding, event)


# ── part 2: the assistant reviews it (free tier) ─────────────────────────────
def will_review(job_id: str, kind: str) -> bool:
    """Whether the model reviews this finding: the job has a chat, the model has tool calling, and the
    job is under its cap (one review per finding kind, MAX_REVIEWS_PER_JOB per job)."""
    if not chat_of(job_id):
        return False
    done = [k for k, f in _load()["findings"].items() if k.startswith(job_id + ":") and f.get("reviewed")]
    if f"{job_id}:{kind}" in done or len(done) >= MAX_REVIEWS_PER_JOB:
        return False
    try:
        return assistant_bridge.tier() == "free"
    except Exception:  # noqa: BLE001 - no session, no review
        return False


def turn_started(chat_id: str | None) -> None:
    if chat_id:
        _busy[chat_id] = _busy.get(chat_id, 0) + 1


def turn_ended(chat_id: str | None) -> None:
    """A user turn in ``chat_id`` ended: start the reviews held for it."""
    if not chat_id:
        return
    _busy[chat_id] = max(0, _busy.get(chat_id, 0) - 1)
    if not _busy[chat_id]:
        for start in _waiting.pop(chat_id, []):
            start()


def _spawn(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def when_free(chat_id: str, start) -> None:
    """Call ``start`` now, or once the turns running in ``chat_id`` have ended."""
    if _busy.get(chat_id):
        _waiting.setdefault(chat_id, []).append(start)              # never while the user has a turn there
    else:
        start()


def _schedule(chat_id: str, job: dict[str, Any], finding: dict[str, Any], event: dict[str, Any]) -> None:
    _update_finding(job["id"], finding["kind"], reviewed=True)       # counts against the cap from now on
    when_free(chat_id, lambda: _spawn(_review(chat_id, job, finding, event)))


class ReviewInteraction:
    """A review has nobody to ask: the OK a stop or relaunch needs becomes an Apply in the chat."""

    def __init__(self, cancel: threading.Event) -> None:
        self.cancel, self.confirming, self.notes = cancel, False, []

    def ask(self, q) -> str:
        from hax_assistant.interaction import Cancelled
        if q.kind == "confirm" and q.key in ("confirm_stop_run", "confirm_relaunch_run") and not self.cancel.is_set():
            self.confirming = True
            return "yes"
        raise Cancelled()

    def notify(self, text: str, level: str = "info") -> None:
        self.notes.append(text)


def _series(workdir: str) -> str:
    from . import watch
    try:
        values = [v for v in watch._loss_series(workdir) if v == v]
    except Exception:  # noqa: BLE001 - the curves are optional context
        return ""
    if not values:
        return ""
    tail, prev = values[-10:], values[-20:-10]
    line = f"training loss: {len(values)} points, last {values[-1]:.4g}, min {min(values):.4g}, mean of the last 10 {sum(tail) / len(tail):.4g}"
    return line + (f", of the 10 before {sum(prev) / len(prev):.4g}" if prev else "")


def _prompt(job: dict[str, Any], finding: dict[str, Any], event: dict[str, Any]) -> str:
    try:
        report = assistant_bridge.watch_report(job["workdir"], True)
        others = "; ".join(f"{f['kind']}: {f['title']}" for f in report["findings"])
    except Exception:  # noqa: BLE001
        others = ""
    rule = event.get("proposal")
    parts = [f"The run monitor flagged the job “{job.get('name') or job['program']}” (job id {job['id']}, run "
             f"folder {job['workdir']}): {finding['title']}. Evidence: {finding.get('evidence')}. The finding is "
             f"{'calibrated' if finding.get('calibrated') else 'not calibrated'}.",
             f"The rule proposes: {rule['text']} (restart {rule['restart']}, changes {rule['changes']})."
             if rule else "The rule proposes no change.",
             f"Measured now: {_series(job['workdir']) or 'no loss curve read'}"
             + (f"; health findings: {others}" if others else "") + ".",
             "Review it with the tools (check_run_health, list_jobs): agree with the rule's change (call relaunch_run "
             "with it), propose a different one (relaunch_run or stop_run with yours), or say to leave it. Say "
             "plainly whether you agree with the rule, and why. Never resume a diverged checkpoint."]
    return "\n".join(parts)


def _compare(rule: dict[str, Any] | None, calls: list[dict[str, Any]]) -> str:
    acts = [c for c in calls if c["name"] in ("relaunch_run", "stop_run")]
    if not acts:
        return ("Compared with the rule: the assistant suggests leaving it as it is"
                + (", unlike the rule's proposal." if rule else "."))
    a = acts[-1]
    if a["name"] == "stop_run":
        return "Compared with the rule: the assistant proposes stopping the run" + (" instead." if rule else ".")
    mine = (a["args"].get("restart"), {k: str(v) for k, v in (a["args"].get("changes") or {}).items()})
    if rule and mine == (rule["restart"], {k: str(v) for k, v in rule["changes"].items()}):
        return "Compared with the rule: the assistant agrees with its proposal."
    return (f"Compared with the rule: the assistant disagrees and proposes {_describe(a['args'])}"
            + (f" instead of {_describe(rule)}." if rule else "."))


def _describe(p: dict[str, Any]) -> str:
    changes = ", ".join(f"--{k} {v}" for k, v in (p.get("changes") or {}).items()) or "no flag changes"
    return f"a {p.get('restart')} restart with {changes}"


def _run_review(messages: list[dict], interaction: ReviewInteraction, tools: list,
                cancel: threading.Event) -> tuple[str, list[dict]]:
    text, calls = [], []
    for ev in assistant_bridge.stream_events(messages, grounded=False, interaction=interaction, tools=tools):
        if cancel.is_set():
            break
        if ev.get("type") == "content":
            text.append(ev.get("delta") or "")
        elif ev.get("type") == "tool":
            calls.append({"name": ev.get("name"), "args": ev.get("args") or {}})
    return "".join(text), calls


async def _review(chat_id: str, job: dict[str, Any], finding: dict[str, Any], event: dict[str, Any]) -> None:
    from . import assistant_tools, watch
    kind, now = finding["kind"], manager.get(job["id"])
    if now is None or now["status"] not in ACTIVE:
        return                                              # the job ended first: nothing to review
    cancel = threading.Event()
    review = {"job_id": job["id"], "kind": kind, "calibrated": bool(finding.get("calibrated")),
              "proposed": None, "acted": None}
    inter = ReviewInteraction(cancel)
    tools = assistant_tools.chat_tools(asyncio.get_running_loop(), inter, cancel, job.get("project_id"),
                                       review=review)
    history = [{"role": m["role"], "content": m["content"]}
               for m in ((chat_store.manager.get(chat_id) or {}).get("messages") or [])[-20:]
               if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    messages = history + [{"role": "user", "content": _prompt(now, finding, event)}]
    worker = asyncio.ensure_future(asyncio.to_thread(_run_review, messages, inter, tools, cancel))
    while not worker.done():
        await asyncio.wait({worker}, timeout=2.0)
        if not worker.done() and (manager.get(job["id"]) or {}).get("status") not in ACTIVE:
            cancel.set()                                    # the job ended meanwhile: the review is dropped
    try:
        text, calls = worker.result()
    except Exception as exc:  # noqa: BLE001 - the finding's own message stays as it is
        _post(chat_id, job["id"], kind, f"The assistant could not review this finding ({exc}).", "review")
        if event.get("deferred"):
            try:
                await watch.apply_proposal(job["id"], kind, by="auto")   # as the watcher would have
            except watch.ApplyError:
                pass
        return
    if cancel.is_set():
        return
    rule = event.get("proposal")
    lines = [text.strip() or "(no review text)", _compare(rule, calls)]
    if review["acted"]:
        lines.append(f"Done automatically: {review['acted']['message']}")
        resolve(job["id"], kind, {**review["acted"], "by": "assistant"})
    elif review["proposed"]:
        lines.append("Waiting for your OK: apply the assistant's change or the rule's below.")
    _post(chat_id, job["id"], kind, "\n\n".join(lines), "review", rule=rule, proposed=review["proposed"],
          resolved=review["acted"])
    _update_finding(job["id"], kind, proposed=review["proposed"])
    if event.get("deferred") and not review["acted"]:
        rec = _finding(job["id"], kind) or {}
        for mid in rec.get("messages", [])[:1]:              # the finding itself: Apply, now that nothing acted
            chat_store.manager.patch_posted(chat_id, mid, deferred=False, needs_confirmation=bool(rule))


async def apply_review(job_id: str, kind: str) -> dict[str, Any]:
    """Carry out the change the assistant's review proposed for a finding, read from the server's record
    (never from the client), with the same checks as the rule's Apply."""
    from . import assistant_tools, watch
    rec = _finding(job_id, kind) or {}
    proposed = rec.get("proposed")
    if not proposed:
        raise watch.ApplyError(404, "no change of the assistant's waits for this finding")
    if rec.get("resolved"):
        raise watch.ApplyError(409, "this finding was already decided")
    job = manager.get(job_id)
    if job is None:
        raise watch.ApplyError(404, "job not found")
    if await asyncio.to_thread(assistant_bridge.decide_relaunch, job["workdir"]) == "propose":
        raise watch.ApplyError(403, "autonomy is advisory (or the run is outside the workspaces): nothing is applied")
    if proposed["tool"] == "stop_run":
        try:
            done = await assistant_tools.stop_job(job_id)
        except assistant_tools.StopError as exc:
            raise watch.ApplyError(exc.status, exc.detail)
        result = {"applied": True, "message": f"stopped “{done['name']}” (its run folder is kept)"}
    else:
        done = await watch.relaunch(job, proposed["restart"], proposed["changes"])
        result = {"applied": done["applied"], "message": done["message"],
                  "new_job_id": (done.get("job") or {}).get("id")}
    if result["applied"]:
        resolve(job_id, kind, {**result, "by": "user"})
    return result


# ── watching a run from a chat (any GUI job, the user's too) ─────────────────
def watch_from_chat(job: dict[str, Any], chat_id: str) -> str:
    """Link a job to a chat and watch it; the one line the chat confirms it with. Runs on the event loop."""
    from . import watch
    before = link(job["id"], chat_id)
    if job["id"] not in watch.watching():
        watch.watch(job["id"])
    name = job.get("name") or job["program"]
    level = assistant_bridge.autonomy_level()
    then = {"advisory": "findings are posted here as proposals; nothing changes without you",
            "ask": "each change the monitor proposes waits for your OK (Apply)",
            "auto": "calibrated problems of the assistant's own jobs are fixed on their own, the rest waits "
                    "for your OK"}[level]
    if not _owned(job):
        then = "you started this run, so every change waits for your OK (Apply), at every autonomy level; " \
               "a relaunch is a new job and your run folder is kept"
    moved = ""
    if before:
        title = (chat_store.manager.get(before) or {}).get("title") or before
        moved = f" (its findings move here from the chat “{title}”)"
    return f"Watching “{name}” ({job['status']}){moved}: {then}."


def unwatch_from_chat(job: dict[str, Any]) -> str:
    from . import watch
    forget_chat(job["id"])
    watch.unwatch(job["id"])
    return f"Stopped watching “{job.get('name') or job['program']}”: no more findings are posted here."


def job_ended(job_id: str) -> None:
    """The watcher saw the job finish: no more findings, so the chat link ends."""
    forget_chat(job_id)


async def restore() -> None:
    """After a restart: watch the linked jobs that still run again, and follow the launched workflows."""
    from . import assistant_workflows, watch
    data = _load()
    for jid, rec in data["jobs"].items():
        job = manager.get(jid)
        if rec.get("chat_id") and job is not None and job["status"] in ACTIVE:
            watch.watch(jid)
    for wid, w in data["workflows"].items():
        assistant_workflows.follow(wid, w["steps"], w["origin"], w.get("chat_id"))
