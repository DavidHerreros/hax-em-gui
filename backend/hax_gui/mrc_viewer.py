"""Map-viewer manager: spawn a per-session ``.mrc`` viewer backend and proxy it.

The volume-viewer counterpart of :mod:`hax_gui.viewer` — instead of a
(latents, network) pair it serves one ``.mrc`` file, so any map a job produced
can be inspected standalone. Each distinct file gets one subprocess (on a private
port); the FastAPI app reverse-proxies it at ``/mrcviewer/<id>/``. The id is an
unguessable capability token (the iframe can't send the bearer header).
"""

from __future__ import annotations

import asyncio
import secrets
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

# The viewer backend ships inside the package: hax_gui/viewer_app/mrc_server.py
_SERVER = Path(__file__).resolve().parent / "viewer_app" / "mrc_server.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MrcViewerManager:
    def __init__(self) -> None:
        self._by_path: dict[str, dict[str, Any]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}

    def get(self, vid: str) -> dict[str, Any] | None:
        return self._by_id.get(vid)

    def list(self) -> list[dict[str, Any]]:
        return [self._public(i) for i in self._by_id.values()]

    async def ensure(self, mrc: str, title: str = "Map") -> dict[str, Any]:
        """Start (or reuse) a viewer for `mrc` and wait until it's up."""
        if not _SERVER.exists():
            raise RuntimeError(f"map-viewer backend not found at {_SERVER}")
        path = Path(mrc).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"map not found: {path}")
        if path.suffix.lower() not in (".mrc", ".map", ".vol"):
            raise ValueError(f"not a volume file: {path}")

        key = str(path)
        existing = self._by_path.get(key)
        if existing and existing["proc"].poll() is None:
            existing["title"] = title
            return self._public(existing)

        vid = secrets.token_urlsafe(9)
        port = _free_port()
        cmd = [
            sys.executable, str(_SERVER),
            "--mrc", str(path),
            "--title", title,
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        instance = {"id": vid, "port": port, "mrc": str(path), "title": title, "proc": proc}
        await self._wait_ready(instance)
        self._by_path[key] = instance
        self._by_id[vid] = instance
        return self._public(instance)

    async def _wait_ready(self, instance: dict[str, Any], timeout: float = 40.0) -> None:
        url = f"http://127.0.0.1:{instance['port']}/api/health"
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient(timeout=3.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                if instance["proc"].poll() is not None:
                    raise RuntimeError("map-viewer backend exited during startup")
                try:
                    r = await client.get(url)
                    if r.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.4)
        raise TimeoutError("map-viewer backend did not become ready in time")

    def stop(self, vid: str) -> bool:
        inst = self._by_id.pop(vid, None)
        if not inst:
            return False
        self._by_path.pop(inst["mrc"], None)
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
            "title": instance["title"],
            "mrc": instance["mrc"],
            "url": f"/mrcviewer/{instance['id']}/",
            "kind": "mrc",
            "running": instance["proc"].poll() is None,
        }


mrc_viewer_manager = MrcViewerManager()
