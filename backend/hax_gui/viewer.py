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

from . import children, viewer_logs

# The viewer backend ships inside the package: hax_gui/viewer_app/server.py
_SERVER = Path(__file__).resolve().parent / "viewer_app" / "server.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ViewerManager:
    def __init__(self) -> None:
        self._by_key: dict[tuple[str, ...], dict[str, Any]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}

    def get(self, vid: str) -> dict[str, Any] | None:
        return self._by_id.get(vid)

    def list(self) -> list[dict[str, Any]]:
        return [self._public(i) for i in self._by_id.values()]

    async def ensure(self, latents: str, reload: str, title: str = "Conformational landscape",
                     metadata: str | None = None, project_dir: str | None = None,
                     reduced: str | None = None, gpu: str | None = None) -> dict[str, Any]:
        """Start (or reuse) a viewer for (latents, reload, gpu) and wait until it's up.

        ``gpu`` pins the subprocess to one device (an index, or "cpu"); None lets
        JAX choose. It is part of the reuse key because the device is fixed at
        spawn — CUDA_VISIBLE_DEVICES is read once, when JAX initialises inside the
        subprocess — so a live viewer cannot be moved. Keying on it means asking
        for a different device starts a second viewer instead of silently handing
        back the one still bound to the old one.
        """
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

        key = (str(lat), str(ckpt), str(red) if red else "", gpu or "")
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
        if gpu:
            cmd += ["--gpu", str(gpu)]
        viewer_logs.prune("viewer")
        log = viewer_logs.log_path("viewer", vid)
        with viewer_logs.spawn_log(log) as log_fh:
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        children.register("viewer", proc.pid)
        instance = {"id": vid, "port": port, "latents": str(lat), "reload": str(ckpt),
                    "title": title, "gpu": gpu or "", "key": key, "proc": proc, "log": str(log)}
        try:
            await self._wait_ready(instance)
        except BaseException:
            # A backend that never came up is not tracked anywhere, so nothing else
            # would ever reap it: kill it here rather than leave it holding a GPU.
            children.release(proc)
            raise
        self._by_key[key] = instance
        self._by_id[vid] = instance
        return self._public(instance)

    async def _wait_ready(self, instance: dict[str, Any], timeout: float = 40.0) -> None:
        url = f"http://127.0.0.1:{instance['port']}/api/health"
        log = Path(instance["log"])
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient(timeout=3.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                if instance["proc"].poll() is not None:
                    raise RuntimeError(viewer_logs.startup_error(
                        "viewer backend exited during startup", log))
                try:
                    r = await client.get(url)
                    if r.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.4)
        raise TimeoutError(viewer_logs.startup_error(
            "viewer backend did not become ready in time", log))

    def stop(self, vid: str) -> bool:
        inst = self._by_id.pop(vid, None)
        if not inst:
            return False
        self._by_key.pop(inst["key"], None)   # the key we actually filed it under
        children.release(inst["proc"])
        return True

    def shutdown(self) -> None:
        for inst in list(self._by_id.values()):
            children.release(inst["proc"])

    @staticmethod
    def _public(instance: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": instance["id"],
            "title": instance["title"],
            "latents": instance["latents"],
            "reload": instance["reload"],
            "gpu": instance["gpu"],   # "" = unpinned (JAX chose); "cpu" = no device
            "url": f"/viewer/{instance['id']}/",
            "log": instance["log"],   # this session's stdout+stderr, for diagnosis
            "running": instance["proc"].poll() is None,
        }


viewer_manager = ViewerManager()
