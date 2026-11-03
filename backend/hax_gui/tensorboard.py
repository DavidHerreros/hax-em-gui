"""TensorBoard manager: launch per-logdir TensorBoard servers behind a subpath.

`display_metrics` in hax launches TensorBoard on a random port and blocks, which
can't be reverse-proxied. Instead we launch TensorBoard ourselves with a fixed
port and ``--path_prefix /tb/<id>`` so the FastAPI app can proxy it to a remote
browser (see the proxy routes in app.py).

The instance id is an unguessable token (it doubles as a capability: the proxy
route can't carry the bearer header because the browser loads it in an iframe).
"""

from __future__ import annotations

import secrets
import socket
import subprocess
import sys
from typing import Any

import httpx

from . import children
from .files import AccessDenied, resolve


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TensorBoardManager:
    def __init__(self) -> None:
        # logdir(str) -> instance dict; reused so we don't spawn duplicates.
        self._by_logdir: dict[str, dict[str, Any]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}
        # ensure() is check-then-spawn; without this two concurrent requests for
        # the same logdir each spawned a TensorBoard and one leaked untracked.
        self._spawn_locks: dict[str, Any] = {}

    def get(self, tb_id: str) -> dict[str, Any] | None:
        return self._by_id.get(tb_id)

    def list(self) -> list[dict[str, Any]]:
        return [self._public(i) for i in self._by_id.values()]

    async def ensure(self, logdir: str) -> dict[str, Any]:
        """Start (or reuse) a TensorBoard server for ``logdir`` and wait until up."""
        target = resolve(logdir)  # sandboxed; raises AccessDenied if outside roots
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"log directory not found: {target}")
        key = str(target)

        async with self._spawn_locks.setdefault(key, _lock()):
            existing = self._by_logdir.get(key)
            if existing and existing["proc"].poll() is None:
                return self._public(existing)

            tb_id = secrets.token_urlsafe(9)
            prefix = f"/tb/{tb_id}"
            last_exc: BaseException | None = None
            # The probed port can be taken between the probe and TB's own bind.
            # TB's output goes to /dev/null, so a startup exit can't be classified —
            # a fresh port is cheap enough to retry on any startup death.
            for _ in range(3):
                port = _free_port()
                proc = subprocess.Popen(
                    [
                        sys.executable, "-m", "tensorboard.main",
                        "--logdir", str(target),
                        "--host", "127.0.0.1",
                        "--port", str(port),
                        "--path_prefix", prefix,
                        # Poll the event files often: a live training run should show up on the
                        # curves promptly, not on TB's 5s-ish default cadence behind a writer
                        # that only flushes periodically.
                        "--reload_interval", "5",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
                children.register("tb", proc.pid)
                instance = {"id": tb_id, "port": port, "prefix": prefix, "logdir": key, "proc": proc}
                try:
                    await self._wait_ready(instance)
                    break
                except BaseException as exc:
                    # Never tracked, so nothing else would ever reap it.
                    children.release(proc)
                    if isinstance(exc, RuntimeError):
                        last_exc = exc
                        continue
                    raise
            else:
                raise last_exc
            self._by_logdir[key] = instance
            self._by_id[tb_id] = instance
            return self._public(instance)

    async def _wait_ready(self, instance: dict[str, Any], timeout: float = 40.0) -> None:
        url = f"http://127.0.0.1:{instance['port']}{instance['prefix']}/"
        deadline = _loop_time() + timeout
        async with httpx.AsyncClient(timeout=3.0) as client:
            while _loop_time() < deadline:
                if instance["proc"].poll() is not None:
                    raise RuntimeError("TensorBoard exited during startup")
                try:
                    r = await client.get(url)
                    if r.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                await _sleep(0.4)
        raise TimeoutError("TensorBoard did not become ready in time")

    def stop(self, tb_id: str) -> bool:
        inst = self._by_id.pop(tb_id, None)
        if not inst:
            return False
        self._by_logdir.pop(inst["logdir"], None)
        children.release(inst["proc"])
        return True

    async def restart(self, tb_id: str) -> dict[str, Any] | None:
        """Respawn the server for this logdir (new process, same logs).

        The projector plugin reads ``projector_config.pbtxt`` once per process and then
        never re-reads it for a JAX logdir — its invalidation check only fires when a
        *TensorFlow* checkpoint changes, and there is none here. So latent-space epochs
        logged after TB started are invisible until TB is restarted. This is that
        restart; hax's own in-UI warning tells users to do it."""
        inst = self._by_id.get(tb_id)
        if inst is None:
            return None
        logdir = inst["logdir"]
        self.stop(tb_id)
        return await self.ensure(logdir)

    def shutdown(self) -> None:
        for inst in list(self._by_id.values()):
            children.release(inst["proc"])

    @staticmethod
    def _public(instance: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": instance["id"],
            "logdir": instance["logdir"],
            "url": instance["prefix"] + "/",
            "running": instance["proc"].poll() is None,
        }


# asyncio helpers kept local so the module has no import-time loop dependency.
import asyncio  # noqa: E402


def _loop_time() -> float:
    return asyncio.get_event_loop().time()


def _lock() -> "asyncio.Lock":
    return asyncio.Lock()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


tb_manager = TensorBoardManager()

# Re-export so callers can catch sandbox errors without importing files.py.
__all__ = ["tb_manager", "AccessDenied"]
