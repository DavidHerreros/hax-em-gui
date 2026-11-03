"""Registry + reaper for GUI-owned child servers (TensorBoard, viewers).

Two failure modes used to leak these subprocesses:

* ``terminate()`` with no ``wait()`` left a zombie per stopped instance, and a
  child that ignored SIGTERM was never escalated to SIGKILL.
* Nothing recorded the children anywhere, so a crashed (or SIGKILLed) server
  left true orphans holding their ports — and, for the landscape viewer, GPU
  memory — with no way to find them afterwards.

Every spawn is therefore recorded in ``children.json`` in the data dir as
``{kind, pid, key}`` where ``key`` is the process start time (see
:mod:`hax_gui.procs`). :func:`release` stops one child with wait + SIGKILL
escalation in a background thread; :func:`reap_orphans`, run at server startup,
kills whatever a previous server left behind — after re-verifying each pid's
identity, so a recycled pid is never signalled.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import threading
import time
from typing import Any

from .config import settings
from .procs import pid_alive, proc_key

_LOCK = threading.Lock()


def _path():
    return settings.data_dir / "children.json"


def _load() -> list[dict[str, Any]]:
    try:
        data = json.loads(_path().read_text())
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save(entries: list[dict[str, Any]]) -> None:
    settings.ensure_dirs()
    tmp = _path().with_name("children.json.tmp")
    try:
        tmp.write_text(json.dumps(entries, indent=2))
        os.replace(tmp, _path())
    except OSError:
        pass


def register(kind: str, pid: int) -> None:
    """Record a freshly spawned child so a later server can find and reap it."""
    key = proc_key(pid)
    with _LOCK:
        entries = [e for e in _load() if e.get("pid") != pid]
        entries.append({"kind": kind, "pid": pid, "key": key})
        _save(entries)


def unregister(pid: int) -> None:
    with _LOCK:
        _save([e for e in _load() if e.get("pid") != pid])


def release(proc: subprocess.Popen) -> None:
    """Stop a tracked child: SIGTERM now, then wait + SIGKILL off-thread.

    The wait matters twice over: without it every stopped instance stayed a
    zombie until the server exited, and a child that ignored SIGTERM (a viewer
    wedged in a CUDA call) was never actually killed. The registry entry is
    dropped only once the child is confirmed gone, so a server crash mid-stop
    still leaves it findable for the next startup's :func:`reap_orphans`.
    """
    try:
        proc.terminate()
    except OSError:
        unregister(proc.pid)
        return
    threading.Thread(target=_wait_kill, args=(proc,), daemon=True).start()


def _wait_kill(proc: subprocess.Popen, grace: float = 5.0) -> None:
    try:
        proc.wait(timeout=grace)
    except Exception:  # noqa: BLE001 - still running (or unwaitable): escalate
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5.0)
    unregister(proc.pid)


def reap_orphans() -> list[dict[str, Any]]:
    """Kill children a previous server process left behind. Blocking (bounded
    ~10 s worst case); run it off the event loop.

    Only pids whose recorded identity still matches are signalled — an entry
    with no key (identity unreadable at spawn) is treated as already gone
    rather than trusted, because SIGKILLing an unverified pid is the worse
    failure.
    """
    with _LOCK:
        entries = _load()
        _save([])
    victims = [e for e in entries
               if e.get("pid") and e.get("key")
               and pid_alive(e["pid"]) and proc_key(e["pid"]) == e["key"]]
    for e in victims:
        with contextlib.suppress(OSError):
            os.kill(e["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not any(pid_alive(e["pid"]) and proc_key(e["pid"]) == e["key"] for e in victims):
            break
        time.sleep(0.2)
    for e in victims:
        if pid_alive(e["pid"]) and proc_key(e["pid"]) == e["key"]:
            with contextlib.suppress(OSError):
                os.kill(e["pid"], signal.SIGKILL)
    return victims
