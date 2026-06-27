"""Command-line control for the hax-em GUI server.

Installed as the ``hax-gui`` console script:

    hax-gui start [--host H] [--port P] [--reload] [--foreground]
    hax-gui stop
    hax-gui restart [--host H] [--port P]
    hax-gui status

``start`` launches uvicorn (serving ``hax_gui.app:app``) detached in the
background and records the pid/host/port in a small state file under the data
dir, so ``stop``/``status``/``restart`` can find it again. Every command prints
whether the server is running.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .config import settings

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


# ── ANSI helpers (degrade to plain text when not a TTY) ──────────────────────
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def _ok(t: str) -> str:    return _c("32", t)
def _warn(t: str) -> str:  return _c("33", t)
def _err(t: str) -> str:   return _c("31", t)
def _bold(t: str) -> str:  return _c("1", t)


# ── state file (pid + bind address) ──────────────────────────────────────────
def _state_path() -> Path:
    return settings.data_dir / "server.json"


def _read_state() -> dict | None:
    try:
        return json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return None


def _write_state(state: dict) -> None:
    settings.ensure_dirs()
    _state_path().write_text(json.dumps(state, indent=2))


def _clear_state() -> None:
    try:
        _state_path().unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists but owned by another user
    return True


def _running() -> dict | None:
    """Return the live server's state, or None (cleaning up a stale file)."""
    state = _read_state()
    if not state:
        return None
    pid = state.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        _clear_state()
        return None
    return state


def _connect_host(host: str) -> str:
    # 0.0.0.0 (all interfaces) isn't directly connectable; probe loopback.
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((_connect_host(host), port), timeout=timeout):
            return True
    except OSError:
        return False


def _health(host: str, port: int, timeout: float = 1.0) -> bool:
    url = f"http://{_connect_host(host)}:{port}/api/health"
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (URLError, OSError, ValueError):
        return False


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_start(args: argparse.Namespace) -> int:
    running = _running()
    if running:
        print(_warn(f"hax-gui is already running") +
              f" — http://{_connect_host(running['host'])}:{running['port']} (pid {running['pid']}).")
        print("Use " + _bold("hax-gui restart") + " to relaunch, or " + _bold("hax-gui stop") + " to stop it.")
        return 1

    host, port = args.host, args.port
    if _port_open(host, port):
        print(_err(f"Port {port} on {host} is already in use") + " by another process. Pick a free --port.")
        return 1

    cmd = [sys.executable, "-m", "uvicorn", "hax_gui.app:app", "--host", host, "--port", str(port)]
    if args.reload:
        cmd.append("--reload")

    if args.foreground:
        print(_bold(f"Starting hax-gui (foreground) on http://{_connect_host(host)}:{port}") + " — Ctrl-C to stop.")
        os.execvp(cmd[0], cmd)   # replace this process; never returns
        return 0                 # unreachable

    # Detached background launch; logs go to the data dir.
    settings.ensure_dirs()
    log = open(settings.data_dir / "server.log", "ab")
    import subprocess
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        start_new_session=True,   # survive the CLI exiting / terminal close
    )
    _write_state({"pid": proc.pid, "host": host, "port": port,
                  "started_at": datetime.now(timezone.utc).isoformat()})

    # Wait briefly for it to actually come up (or fail fast).
    deadline = time.time() + 12
    while time.time() < deadline:
        if proc.poll() is not None:
            _clear_state()
            print(_err("hax-gui failed to start.") + f" See {settings.data_dir / 'server.log'}")
            return 1
        if _health(host, port):
            print(_ok("hax-gui is running") + f" — http://{_connect_host(host)}:{port} (pid {proc.pid}).")
            print(f"Logs: {settings.data_dir / 'server.log'}   Stop: " + _bold("hax-gui stop"))
            return 0
        time.sleep(0.3)

    print(_warn("hax-gui started but did not answer /api/health yet") +
          f" — http://{_connect_host(host)}:{port} (pid {proc.pid}). Check " + str(settings.data_dir / "server.log") + ".")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    running = _running()
    if not running:
        print(_warn("hax-gui is not running."))
        return 0
    pid = running["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_state()
        print(_warn("hax-gui is not running."))
        return 0

    # Wait for a graceful exit, then force-kill.
    for _ in range(40):   # ~10s
        if not _pid_alive(pid):
            break
        time.sleep(0.25)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _clear_state()
    print(_ok("hax-gui stopped.") + f" (pid {pid})")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    running = _running()
    # Preserve the current bind address unless the user overrode it explicitly.
    if running:
        if args.host == DEFAULT_HOST and not args._host_set:
            args.host = running["host"]
        if args.port == DEFAULT_PORT and not args._port_set:
            args.port = running["port"]
        cmd_stop(args)
    else:
        print(_warn("hax-gui was not running; starting it."))
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:
    running = _running()
    if not running:
        print(_err("● hax-gui is not running."))
        print("Start it with " + _bold("hax-gui start") + ".")
        return 1
    host, port, pid = running["host"], running["port"], running["pid"]
    url = f"http://{_connect_host(host)}:{port}"
    healthy = _health(host, port)
    dot = _ok("●") if healthy else _warn("●")
    state = _ok("running") if healthy else _warn("running (not answering /api/health yet)")
    print(f"{dot} hax-gui is {state}.")
    print(f"    url:     {url}")
    print(f"    pid:     {pid}")
    if running.get("started_at"):
        print(f"    started: {running['started_at']}")
    print(f"    logs:    {settings.data_dir / 'server.log'}")
    return 0 if healthy else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hax-gui", description="Control the hax-em GUI server.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_bind(sp):
        sp.add_argument("--host", default=DEFAULT_HOST, help=f"bind address (default {DEFAULT_HOST})")
        sp.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})")

    sp = sub.add_parser("start", help="start the server in the background")
    add_bind(sp)
    sp.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    sp.add_argument("--foreground", "-f", action="store_true", help="run attached to this terminal instead of detaching")
    sp.set_defaults(func=cmd_start)

    sp = sub.add_parser("stop", help="stop the running server")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("restart", help="restart the server (keeps host/port unless overridden)")
    add_bind(sp)
    sp.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    sp.add_argument("--foreground", "-f", action="store_true", help="run attached to this terminal instead of detaching")
    sp.set_defaults(func=cmd_restart)

    sp = sub.add_parser("status", help="show whether the server is running")
    sp.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # Track whether --host/--port were explicitly given (so restart can preserve them).
    parser = build_parser()
    args = parser.parse_args(argv)
    args._host_set = any(a == "--host" or a.startswith("--host=") for a in argv)
    args._port_set = any(a == "--port" or a.startswith("--port=") for a in argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
