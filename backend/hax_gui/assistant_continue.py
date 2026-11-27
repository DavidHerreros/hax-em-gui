"""The chat continues when a run it launched ends.

When the chat model launches a job or workflow it may leave a note for itself (``then``): what to do
once it has succeeded. When it succeeds, at autonomy auto (and a tool-calling model) the server starts
that turn in the same chat, as it starts a monitor review: the same tools and gates, nobody is asked,
and what it did is posted there. Otherwise the chat gets a Continue button that sends the note as the
user's own message, so every OK is asked as usual. A failure is posted and nothing continues; after
MAX_CHAIN automatic turns in a row the next one needs the button. Kept in
``<data_dir>/assistant_continue.json`` and re-armed at startup.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any

from . import assistant as assistant_bridge
from . import assistant_monitor, events, jsonfile
from . import chats as chat_store
from .config import settings
from .jobs import ACTIVE, manager

FILE = "assistant_continue.json"
MAX_CHAIN = 5
POLL_S = 3.0
WF_RUNNING = ("running", "idle", "interrupted")
_lock = threading.Lock()
_tasks: set[asyncio.Task] = set()


def _path():
    return settings.data_dir / FILE


def _load() -> dict[str, Any]:
    data = jsonfile.read(_path(), {})
    return data if isinstance(data, dict) else {}


def _change(key: str, rec: dict[str, Any] | None) -> None:
    settings.ensure_dirs()
    with _lock:
        data = _load()
        if rec is None:
            data.pop(key, None)
        else:
            data[key] = rec
        jsonfile.write(_path(), data)


def _spawn(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def arm(kind: str, target_id: str, chat_id: str, note: str, depth: int, project_id: str | None) -> None:
    """Continue ``chat_id`` with ``note`` once the job or workflow ``target_id`` has succeeded."""
    key = f"{kind}:{target_id}"
    _change(key, {"kind": kind, "id": target_id, "chat_id": chat_id, "note": note.strip(), "depth": depth,
                  "project_id": project_id, "armed": time.time()})
    _spawn(_wait(key))


async def restore() -> None:
    """After a restart: wait again for every run a chat is to continue after."""
    for key in _load():
        _spawn(_wait(key))


def _state(rec: dict[str, Any]) -> tuple[str | None, str]:
    """The run's status (None: it no longer exists) and how the chat names it."""
    if rec["kind"] == "job":
        job = manager.get(rec["id"])
        if job is None:
            return None, ""
        return job["status"], f"the job “{job.get('name') or job['program']}” (run folder {job.get('workdir')})"
    from .workflows import manager as wf_manager
    wf = wf_manager.get(rec["id"])
    if wf is None:
        return None, ""
    return wf["status"], f"the workflow “{wf['name']}”"


def _ended(rec: dict[str, Any], status: str) -> bool:
    return status not in ACTIVE if rec["kind"] == "job" else status not in WF_RUNNING


async def _wait(key: str) -> None:
    while True:
        rec = _load().get(key)
        if rec is None:
            return
        status, what = _state(rec)
        if status is None or _ended(rec, status):
            break
        await asyncio.sleep(POLL_S)
    _change(key, None)
    if status is None or chat_store.manager.get(rec["chat_id"]) is None:
        return
    if status != "succeeded":
        _post(rec["chat_id"], f"**Next step not started** — {what} ended as {status}, so the assistant did not go "
                              f"on with: {rec['note']}")
        return
    prompt = (f"[automatic: a run you launched has finished] {what} has succeeded. Your note for this moment: "
              f"{rec['note']}\nGo on with it now, with your tools (list_jobs shows the jobs and their run folders), "
              "and say briefly what you did.")
    auto = assistant_bridge.autonomy_level() == "auto" and rec["depth"] < MAX_CHAIN
    try:
        auto = auto and assistant_bridge.tier() == "free"
    except Exception:  # noqa: BLE001 - no model: the user continues
        auto = False
    if auto:
        assistant_monitor.when_free(rec["chat_id"], lambda: _spawn(_turn(rec, prompt)))
        return
    why = (f" It went on by itself {MAX_CHAIN} times in a row, so this step waits for you."
           if rec["depth"] >= MAX_CHAIN and assistant_bridge.autonomy_level() == "auto" else "")
    _post(rec["chat_id"], f"**Next step** — {what} has succeeded. The assistant's plan for now: {rec['note']}{why}",
          next={"prompt": prompt, "shown": f"Continue: {rec['note']}", "used": False})


class _Nobody:
    """A continuation has nobody to ask: anything that needs an OK is declined and the model says so."""

    def __init__(self) -> None:
        self.notes: list[str] = []

    def ask(self, q) -> str:
        from hax_assistant.interaction import Cancelled
        raise Cancelled()

    def notify(self, text: str, level: str = "info") -> None:
        self.notes.append(text)


async def _turn(rec: dict[str, Any], prompt: str) -> None:
    from . import assistant_tools
    chat_id = rec["chat_id"]
    assistant_monitor.turn_started(chat_id)          # reviews and other continuations wait for it
    try:
        cancel, inter = threading.Event(), _Nobody()
        tools = assistant_tools.chat_tools(asyncio.get_running_loop(), inter, cancel, rec.get("project_id"),
                                           chat_id=chat_id, depth=rec["depth"] + 1)
        history = [{"role": m["role"], "content": m["content"]}
                   for m in ((chat_store.manager.get(chat_id) or {}).get("messages") or [])[-20:]
                   if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        try:
            text, _ = await asyncio.to_thread(assistant_monitor._run_review,
                                              history + [{"role": "user", "content": prompt}], inter, tools, cancel)
        except Exception as exc:  # noqa: BLE001 - the user can still go on by hand
            _post(chat_id, f"The assistant could not go on by itself ({exc}). Its plan was: {rec['note']}",
                  next={"prompt": prompt, "shown": f"Continue: {rec['note']}", "used": False})
            return
        _post(chat_id, "\n\n".join([text.strip() or "(no reply)"] + inter.notes))
    finally:
        assistant_monitor.turn_ended(chat_id)


def _post(chat_id: str, text: str, **card: Any) -> None:
    msg = {"role": "assistant", "content": text, "posted": "continue", "id": uuid.uuid4().hex[:12], **card}
    if chat_store.manager.post(chat_id, msg) is not None:
        events.publish({"kind": "chat", "chat_id": chat_id})


def mark_used(chat_id: str, mid: str) -> dict[str, Any] | None:
    """The Continue button of a posted next step was clicked: it is not offered again."""
    msg = next((m for m in (chat_store.manager.get(chat_id) or {}).get("messages") or []
                if isinstance(m, dict) and m.get("id") == mid and m.get("posted") == "continue" and m.get("next")
                and not m["next"].get("used")), None)
    if msg is None:
        return None
    return chat_store.manager.patch_posted(chat_id, mid, next={**msg["next"], "used": True})
