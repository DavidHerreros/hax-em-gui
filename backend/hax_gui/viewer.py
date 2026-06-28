"""WebGL viewer manager: spawn a per-session viewer backend and proxy it.

Each session runs the bundled viewer backend (in this same hax env) for one
(latents, network) pair on a private port; the FastAPI app reverse-proxies it at
``/viewer/<id>/`` (see app.py). The frontend uses relative URLs, so the proxy
just strips the ``/viewer/<id>`` prefix and forwards the rest to the subprocess
root. The id is an unguessable capability token (the iframe can't send the
bearer header).
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

# The viewer backend ships inside the package: hax_gui/viewer_app/server.py
_SERVER = Path(__file__).resolve().parent / "viewer_app" / "server.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ViewerManager:
    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], dict[str, Any]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}

    def get(self, vid: str) -> dict[str, Any] | None:
        return self._by_id.get(vid)

    def list(self) -> list[dict[str, Any]]:
        return [self._public(i) for i in self._by_id.values()]

    async def ensure(self, latents: str, reload: str, title: str = "Conformational landscape",
                     metadata: str | None = None, project_dir: str | None = None,
                     reduced: str | None = None) -> dict[str, Any]:
        """Start (or reuse) a viewer for (latents, reload) and wait until it's up."""
        if not _SERVER.exists():
            raise RuntimeError(f"viewer backend not found at {_SERVER}")
        lat = Path(latents).expanduser()
        ckpt = Path(reload).expanduser()
        if not lat.exists():
            raise FileNotFoundError(f"latents file not found: {lat}")
        if not (ckpt / "checkpoints").is_dir():
            raise FileNotFoundError(f"network checkpoint not found at: {ckpt}")
        red = Path(reduced).expanduser() if reduced else None
        if red is not None and not red.exists():
            red = None   # ignore a missing reduced-space file rather than failing the launch

        key = (str(lat), str(ckpt), str(red) if red else "")
        existing = self._by_key.get(key)
        if existing and existing["proc"].poll() is None:
            existing["title"] = title
            return self._public(existing)

        vid = secrets.token_urlsafe(9)
        port = _free_port()
        cmd = [
            sys.executable, str(_SERVER),
            "--latents", str(lat),
            "--reload", str(ckpt),
            "--title", title,
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        if red is not None:
            cmd += ["--z_space_reduced", str(red)]
        if project_dir:
            cmd += ["--project-dir", str(project_dir)]
        if metadata:
            cmd += ["--metadata", str(metadata)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        instance = {"id": vid, "port": port, "latents": str(lat), "reload": str(ckpt), "title": title, "proc": proc}
        await self._wait_ready(instance)
        self._by_key[key] = instance
        self._by_id[vid] = instance
        return self._public(instance)

    async def _wait_ready(self, instance: dict[str, Any], timeout: float = 40.0) -> None:
        url = f"http://127.0.0.1:{instance['port']}/api/health"
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient(timeout=3.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                if instance["proc"].poll() is not None:
                    raise RuntimeError("viewer backend exited during startup")
                try:
                    r = await client.get(url)
                    if r.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.4)
        raise TimeoutError("viewer backend did not become ready in time")

    def stop(self, vid: str) -> bool:
        inst = self._by_id.pop(vid, None)
        if not inst:
            return False
        self._by_key.pop((inst["latents"], inst["reload"]), None)
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
            "latents": instance["latents"],
            "reload": instance["reload"],
            "url": f"/viewer/{instance['id']}/",
            "running": instance["proc"].poll() is None,
        }


viewer_manager = ViewerManager()
