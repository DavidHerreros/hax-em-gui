"""Optional lifecycle management for a local Ollama server.

The GUI may start Ollama itself when it isn't already running, and — matching
that ownership — stop only the Ollama *it* started when the GUI shuts down. An
Ollama the user launched independently (already up before us) is never touched.

Ownership survives an in-place server restart via a pidfile in the data dir:
after a restart the new process finds Ollama still up with a live, GUI-written
pid and reclaims ownership, so the eventual real shutdown still cleans it up.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from .config import settings


def _pidfile() -> Path:
    return settings.data_dir / "ollama.pid"


def _host_root(base_url: str) -> str:
    """``http://host:port`` from an Ollama base url (drops a trailing ``/v1``)."""
    u = urlparse(base_url)
    scheme = u.scheme or "http"
    host = u.hostname or "localhost"
    port = u.port or 11434
    return f"{scheme}://{host}:{port}"


def is_local(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def is_running(base_url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(_host_root(base_url) + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class OllamaManager:
    """Tracks whether *this* GUI owns a local Ollama process."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._owned = False

    @property
    def owned(self) -> bool:
        return self._owned

    def ensure_started(self, base_url: str, wait: float = 20.0) -> dict[str, Any]:
        """Start Ollama if it is local and not already reachable. Best-effort;
        returns a small status dict describing what happened."""
        if not is_local(base_url):
            return {"managed": False, "reason": "remote Ollama"}
        pf = _pidfile()
        if is_running(base_url):
            # already up — reclaim ownership if a prior (restarted) GUI started it
            pid = _read_pid(pf)
            if pid and _pid_alive(pid):
                self._owned = True
                return {"managed": True, "started": False, "reclaimed": True, "pid": pid}
            _unlink(pf)  # stale or foreign
            return {"managed": False, "started": False, "reason": "already running"}
        try:
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # own process group, so we can signal the tree
            )
        except FileNotFoundError:
            return {"managed": False, "started": False, "reason": "ollama not on PATH"}
        self._proc = proc
        self._owned = True
        settings.ensure_dirs()
        _write_pid(pf, proc.pid)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if is_running(base_url):
                return {"managed": True, "started": True, "pid": proc.pid}
            if proc.poll() is not None:  # died on startup
                self._owned = False
                _unlink(pf)
                return {"managed": False, "started": False, "reason": "ollama serve exited"}
            time.sleep(0.4)
        return {"managed": True, "started": True, "pid": proc.pid, "reachable": False}

    def stop_if_owned(self) -> bool:
        """Stop Ollama only if this GUI started it (or reclaimed a prior GUI's).
        Never touches a user-run Ollama. Returns True if it signalled a process."""
        if not self._owned:
            return False
        pf = _pidfile()
        proc = self._proc
        pid = proc.pid if proc is not None else _read_pid(pf)
        stopped = False
        if pid and _pid_alive(pid):
            _terminate(pid)
            stopped = True
        if proc is not None:  # reap our direct child so it doesn't linger as a zombie
            try:
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001 - already gone or slow; best-effort
                pass
        self._owned = False
        self._proc = None
        _unlink(pf)
        return stopped

    def status(self, base_url: str) -> dict[str, Any]:
        return {
            "local": is_local(base_url),
            "running": is_running(base_url),
            "owned": self._owned,
        }


def _read_pid(pf: Path) -> Optional[int]:
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def _write_pid(pf: Path, pid: int) -> None:
    try:
        pf.write_text(str(pid))
    except OSError:
        pass


def _unlink(pf: Path) -> None:
    try:
        pf.unlink(missing_ok=True)
    except OSError:
        pass


def _terminate(pid: int) -> None:
    """SIGTERM the process group, escalate to SIGKILL if it lingers."""
    def _signal(sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    _signal(signal.SIGTERM)
    for _ in range(20):  # up to ~2s for a graceful exit
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    _signal(signal.SIGKILL)


manager = OllamaManager()
