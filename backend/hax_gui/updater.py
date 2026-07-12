"""One-click in-GUI updates for ``hax-em`` / ``hax-em-gui`` (Phase 2).

Runs an update *plan* — one or more commands streamed live in sequence, then a
final action — reusing the same chunk pub/sub shape as :mod:`hax_gui.jobs` so the
frontend renders it in the very same terminal component.

The plan depends on the detected install kind and, for editable (devel) installs,
on the user's chosen ``mode``:

* **pypi**     → ``<python> -m pip install -U <dist>[extra]``           → restart if GUI
* **vcs**      → ``<python> -m pip install -U 'git+<url>'``             → restart if GUI
* **editable, mode="full"** → ``git pull --ff-only`` [+ ``pip install -e <repo>[extra]``
  when pyproject.toml changed]                                          → restart if GUI
* **editable, mode="pull"** → ``git pull --ff-only`` only              → *close* the GUI
  server (not restart) so the user installs dependencies by hand, then ``hax-gui start``

Dependency changes are never resolved automatically in ``pull`` mode — a devel
pull can shift the tightly-pinned JAX/CUDA stack, so the user is shown the exact
commands (with the detected CUDA extra) and the server steps aside. Only one
update runs at a time, and never while a job is active.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import sys
from collections import deque
from typing import Any

from . import updates
from .jobs import manager as job_manager

LIVE_BUFFER_CHARS = 64_000
# git must never block on a credential/passphrase prompt for an editable pull.
_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


class UpdateBusy(Exception):
    """An update is already running."""


class UpdateBlocked(Exception):
    """Refused for a stated reason (active jobs, dirty tree, unknown kind…)."""


def _pip(*args: str) -> list[str]:
    # Act on *this* server's environment, not whatever `pip` is on PATH.
    return [sys.executable, "-m", "pip", "install", *args]


def _spec(target: str, extra: str | None) -> str:
    # argv form: brackets are literal, no shell quoting needed.
    return f"{target}[{extra}]" if extra else target


class Updater:
    """Single-flight update runner with a live chunk stream."""

    def __init__(self) -> None:
        self._status = "idle"          # idle | running | succeeded | failed
        self._package: str | None = None
        self._return_code: int | None = None
        self._restarting = False
        self._closing = False
        self._note: str | None = None
        self._buffer: deque[str] = deque()
        self._buflen = 0
        self._subscribers: set[asyncio.Queue] = set()

    # -- state -------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "package": self._package,
            "return_code": self._return_code,
            "restarting": self._restarting,
            "closing": self._closing,
            "note": self._note,
            "active": self._status == "running",
        }

    # -- launch ------------------------------------------------------------
    async def run(self, package: str, mode: str = "full") -> dict[str, Any]:
        """Validate, gate, then launch the update plan for ``package`` (returns
        once the first command has started; output streams over the pub/sub)."""
        if self._status == "running":
            raise UpdateBusy()
        if mode not in ("full", "pull"):
            raise UpdateBlocked(f"unknown update mode '{mode}'")
        if not updates.is_watched(package):
            raise UpdateBlocked(f"'{package}' is not a managed package")
        # Updating the GUI no longer disturbs a running job: jobs own their log files and
        # their own process group, so the restart at the end of the update leaves them
        # alone and the next server process reattaches to them (see jobs.py).
        #
        # Updating hax-em itself is a different matter, and still blocked. A running job
        # *is* hax: pip would be replacing the very modules it is executing, and it
        # imports lazily as it goes, so the files could change underneath it mid-run.
        if package != updates.GUI_DIST and job_manager.count_active() > 0:
            raise UpdateBlocked(
                f"a job is still running — '{package}' is the package those jobs execute, "
                "so updating it now could change the code underneath them. Wait for the job "
                "to finish, or cancel it."
            )

        status = await asyncio.to_thread(updates.status_for, package)
        if status is None:
            raise UpdateBlocked(f"'{package}' is not installed")
        plan = self._plan(status, mode)

        self._package = package
        self._status = "running"
        self._return_code = None
        self._restarting = False
        self._closing = False
        self._note = None
        self._buffer = deque()
        self._buflen = 0
        asyncio.create_task(self._execute(plan))
        return self.state()

    def _plan(self, status: dict[str, Any], mode: str) -> dict[str, Any]:
        """Turn a package status into ``{steps, final}``. Raises UpdateBlocked."""
        kind = status.get("kind")
        dist = status["name"]
        is_gui = status.get("is_gui", False)
        extra = status.get("extra")
        steps: list[tuple[str, list[str], str | None]] = []

        if kind == "editable":
            repo = status.get("repo")
            if not repo:
                raise UpdateBlocked("could not locate the source checkout")
            if status.get("dirty"):
                raise UpdateBlocked(
                    "the source tree has uncommitted changes — commit or stash them, then update"
                )
            # --ff-only: never create a merge commit or stall on conflicts.
            steps.append(("Pulling latest changes", ["git", "-C", repo, "pull", "--ff-only"], repo))
            if mode == "pull":
                # Bring the code; the user installs deps and restarts by hand. For
                # the GUI we step aside (close); hax-em can be updated live.
                final = "stop" if is_gui else "none"
            else:  # full
                if status.get("pyproject_changed"):
                    steps.append(("Installing dependencies", _pip("-e", _spec(repo, extra)), repo))
                final = "restart" if is_gui else "none"
        elif kind == "vcs":
            url = status.get("vcs_url")
            target = f"git+{url}" if url else dist
            steps.append(("Reinstalling from git", _pip("-U", target), None))
            final = "restart" if is_gui else "none"
        elif kind == "pypi":
            steps.append(("Upgrading from PyPI", _pip("-U", _spec(dist, extra)), None))
            final = "restart" if is_gui else "none"
        else:
            raise UpdateBlocked(f"don't know how to update a '{kind}' install")
        return {"steps": steps, "final": final}

    # -- execution ---------------------------------------------------------
    async def _execute(self, plan: dict[str, Any]) -> None:
        for label, argv, cwd in plan["steps"]:
            self._publish({"type": "chunk", "data": f"\n[{label}]\n$ {' '.join(argv)}\n"})
            code = await self._stream_proc(argv, cwd)
            if code != 0:
                self._finish("failed", code, note=f"{label.lower()} failed — see the log above")
                return

        final = plan["final"]
        if final == "restart":
            await self._finish_with_restart()
        elif final == "stop":
            await self._finish_with_stop()
        else:
            self._finish("succeeded", 0, note="Update installed. New runs will use it.")

    async def _stream_proc(self, argv: list[str], cwd: str | None) -> int:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=cwd,
                env=_GIT_ENV,
            )
        except OSError as exc:
            self._append(f"could not launch: {exc}\n")
            self._publish({"type": "chunk", "data": f"could not launch: {exc}\n"})
            return -1

        assert proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            data = await proc.stdout.read(4096)
            if not data:
                break
            text = decoder.decode(data)
            if text:
                self._append(text)
                self._publish({"type": "chunk", "data": text})
        return await proc.wait()

    async def _finish_with_restart(self) -> None:
        from . import cli

        if cli.spawn_restart():
            self._publish({"type": "chunk", "data": "\nUpdate installed — restarting the server…\n"})
            self._finish("succeeded", 0, restarting=True,
                         note="The server is restarting to load the new version.")
        else:
            self._finish("succeeded", 0, note=(
                "Update installed. Restart the server to load it "
                "(the GUI wasn't started with `hax-gui start`, so it can't self-restart)."
            ))

    async def _finish_with_stop(self) -> None:
        from . import cli

        if cli.spawn_stop():
            self._publish({"type": "chunk", "data": (
                "\nChanges pulled — closing the server so you can update dependencies cleanly.\n"
            )})
            self._finish("succeeded", 0, closing=True, note=(
                "The server is closing. Run the dependency command, then start it again."
            ))
        else:
            self._finish("succeeded", 0, note=(
                "Changes pulled. Install any new dependencies, then restart the server "
                "(it wasn't started with `hax-gui start`, so it can't close itself)."
            ))

    def _finish(self, status: str, code: int, *, restarting: bool = False,
                closing: bool = False, note: str | None = None) -> None:
        self._status = status
        self._return_code = code
        self._restarting = restarting
        self._closing = closing
        self._note = note
        self._publish({"type": "status", "status": status, "return_code": code,
                       "restarting": restarting, "closing": closing, "note": note})

    # -- pub/sub (mirrors JobManager) --------------------------------------
    def _append(self, text: str) -> None:
        self._buffer.append(text)
        self._buflen += len(text)
        while self._buflen > LIVE_BUFFER_CHARS and len(self._buffer) > 1:
            self._buflen -= len(self._buffer.popleft())

    def _publish(self, message: dict[str, Any]) -> None:
        for queue in self._subscribers:
            queue.put_nowait(message)

    def subscribe(self) -> tuple[asyncio.Queue, str]:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue, "".join(self._buffer)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)


updater = Updater()
