"""Map-viewer manager: spawn a per-session ``.mrc`` viewer backend and proxy it.

The volume-viewer counterpart of :mod:`hax_gui.viewer` — instead of a
(latents, network) pair it serves ``.mrc`` files, so any map a job produced can be
inspected standalone. One request may name several maps, which then share a single
viewer: one scene, one camera, one model panel, and a morph between them. Each
distinct *set* of maps gets one subprocess (on a private port); the FastAPI app
reverse-proxies it at ``/mrcviewer/<id>/``. The id is an unguessable capability
token (the iframe can't send the bearer header).
"""

from __future__ import annotations

import asyncio
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from . import children, viewer_logs

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
        # ensure() is check-then-spawn; without this a double-click spawned two
        # subprocesses for the same key and leaked one (see viewer.py).
        self._spawn_locks: dict[str, asyncio.Lock] = {}

    def get(self, vid: str) -> dict[str, Any] | None:
        return self._by_id.get(vid)

    def touch(self, vid: str) -> None:
        inst = self._by_id.get(vid)
        if inst is not None:
            inst["last_seen"] = time.time()

    def reap_idle(self, max_idle_s: float) -> list[str]:
        """Stop sessions with no proxied request for ``max_idle_s``; drop dead ones."""
        stopped = []
        for vid, inst in list(self._by_id.items()):
            if inst["proc"].poll() is not None:
                self.stop(vid)
            elif time.time() - inst.get("last_seen", 0) > max_idle_s:
                self.stop(vid)
                stopped.append(vid)
        return stopped

    def list(self) -> list[dict[str, Any]]:
        return [self._public(i) for i in self._by_id.values()]

    async def ensure(self, mrc: str | list[str], title: str = "Map") -> dict[str, Any]:
        """Start (or reuse) a viewer for one map, or for several shown together.

        Several maps go into a *single* viewer — one scene, one camera, one model
        panel — rather than one tab each, which is the whole point of opening them
        together. The reuse key is the ordered tuple of paths, so asking for the same
        set again returns the running viewer while asking for a different set (or the
        same maps in a different order) starts its own.
        """
        if not _SERVER.exists():
            raise RuntimeError(f"map-viewer backend not found at {_SERVER}")
        raw = [mrc] if isinstance(mrc, str) else list(mrc)
        if not raw:
            raise ValueError("no map given")
        paths: list[Path] = []
        for one in raw:
            path = Path(one).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"map not found: {path}")
            if path.suffix.lower() not in (".mrc", ".map", ".vol", ".ccp4"):
                raise ValueError(f"not a volume file: {path}")
            paths.append(path)

        key = "\n".join(str(p) for p in paths)
        async with self._spawn_locks.setdefault(key, asyncio.Lock()):
            existing = self._by_path.get(key)
            if existing and existing["proc"].poll() is None:
                existing["title"] = title
                return self._public(existing)

            vid = secrets.token_urlsafe(9)
            viewer_logs.prune("mrcviewer")
            log = viewer_logs.log_path("mrcviewer", vid)
            last_exc: BaseException | None = None
            for _ in range(3):
                port = _free_port()
                cmd = [
                    sys.executable, str(_SERVER),
                    "--mrc", *[str(p) for p in paths],
                    "--title", title,
                    "--host", "127.0.0.1",
                    "--port", str(port),
                ]
                with viewer_logs.spawn_log(log) as log_fh:
                    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
                children.register("mrcviewer", proc.pid)
                instance = {"id": vid, "port": port, "mrcs": [str(p) for p in paths],
                            "key": key, "title": title, "proc": proc, "log": str(log),
                            "last_seen": time.time()}
                try:
                    await self._wait_ready(instance)
                    break
                except BaseException as exc:
                    # Untracked once we raise, so nothing else would reap it.
                    children.release(proc)
                    if isinstance(exc, RuntimeError) and viewer_logs.bind_failure(Path(log)):
                        last_exc = exc     # port lost to the bind race: retry fresh
                        continue
                    raise
            else:
                raise last_exc
            self._by_path[key] = instance
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
                        "map-viewer backend exited during startup", log))
                try:
                    r = await client.get(url)
                    if r.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.4)
        raise TimeoutError(viewer_logs.startup_error(
            "map-viewer backend did not become ready in time", log))

    def stop(self, vid: str) -> bool:
        inst = self._by_id.pop(vid, None)
        if not inst:
            return False
        self._by_path.pop(inst["key"], None)   # the key we actually filed it under
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
            # `mrc` stays for one-map callers; `mrcs` is the full list.
            "mrc": instance["mrcs"][0],
            "mrcs": instance["mrcs"],
            "url": f"/mrcviewer/{instance['id']}/",
            "log": instance["log"],   # this session's stdout+stderr, for diagnosis
            "kind": "mrc",
            "running": instance["proc"].poll() is None,
        }


mrc_viewer_manager = MrcViewerManager()
