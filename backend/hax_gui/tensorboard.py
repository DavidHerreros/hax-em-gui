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

        existing = self._by_logdir.get(key)
        if existing and existing["proc"].poll() is None:
            return self._public(existing)

        tb_id = secrets.token_urlsafe(9)
        port = _free_port()
        prefix = f"/tb/{tb_id}"
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "tensorboard.main",
                "--logdir", str(target),
                "--host", "127.0.0.1",
                "--port", str(port),
                "--path_prefix", prefix,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        instance = {"id": tb_id, "port": port, "prefix": prefix, "logdir": key, "proc": proc}
        await self._wait_ready(instance)
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
        try:
            inst["proc"].terminate()
        except OSError:
            pass
        return True

    def shutdown(self) -> None:
        for inst in list(self._by_id.values()):
            try:
                inst["proc"].terminate()
            except OSError:
                pass

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


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


tb_manager = TensorBoardManager()

# Re-export so callers can catch sandbox errors without importing files.py.
__all__ = ["tb_manager", "AccessDenied"]
