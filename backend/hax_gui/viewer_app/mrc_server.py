"""Standalone backend for the WebGL *map* viewer (a single ``.mrc`` volume).

This is the volume-viewer half of the conformational-landscape app, extracted so
any map a job produced (during or after a run) can be inspected on its own — no
network, no latents, no landscape. It loads one ``.mrc`` into a NumPy grid and
serves it as an isosurface mesh (marching cubes), a 3D-texture payload (for the
ray-marcher) and orthogonal slices, reusing the same helpers and frontend
renderers as ``server.py``:

    python mrc_server.py --mrc /path/map.mrc --port 8052

It is also spawned per-session by the main GUI backend, which reverse-proxies it
at ``/mrcviewer/<id>/`` (the frontend uses relative URLs, so it works under any
prefix).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Be a good GPU citizen even though this backend never touches the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Run as a plain script (the GUI spawns `python mrc_server.py`), so the sibling
# `server.py` is importable as a top-level module for the shared mesh/grid helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server import _downsample, level_for_mass, level_for_sigma, mesh_at  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"


def _load_mrc(path: str) -> np.ndarray:
    """Read a .mrc/.vol into a writable float32 3D grid (via xmipp ImageHandler)."""
    from xmipp_metadata.image_handler import ImageHandler

    vol = np.asarray(ImageHandler(path).getData(), dtype=np.float32)
    vol = np.squeeze(vol)
    if vol.ndim == 4:
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"{path} is not a 3D volume (got shape {vol.shape})")
    # ImageHandler may hand back a read-only view; mesh_at / slicing need a copy.
    return np.ascontiguousarray(vol, dtype=np.float32)


def build_app(volume: np.ndarray, title: str = "Map", filename: str = "") -> FastAPI:
    app = FastAPI(title="Hax WebGL map viewer")
    vol = volume
    stats = {
        "min": float(vol.min()), "max": float(vol.max()),
        "mean": float(vol.mean()), "std": float(vol.std()),
    }

    # The frontend is plain ES modules served statically; never cache them so code
    # changes take effect on reload (mirrors server.py).
    @app.middleware("http")
    async def _no_cache(request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "shape": list(vol.shape)}

    @app.get("/api/info")
    def info() -> dict:
        return {
            "title": title,
            "filename": filename,
            "shape": [int(s) for s in vol.shape],   # (nz, ny, nx) in NumPy order
            **stats,
            "sigma_level": level_for_sigma(vol, 2.5),
        }

    @app.get("/api/histogram")
    def histogram(bins: int = Query(96)) -> dict:
        counts, edges = np.histogram(vol.ravel(), bins=int(max(8, min(256, bins))))
        return {
            "counts": counts.astype(int).tolist(),
            "edges": edges.astype(float).tolist(),
            **stats,
            "sigma_level": level_for_sigma(vol, 2.5),
        }

    @app.get("/api/decode")
    def decode(
        sigma: float = Query(2.5),
        mass: float = Query(-1.0),     # >=0 -> contour at constant enclosed mass
        smooth: int = Query(0),        # Taubin smoothing iterations on the surface
    ) -> Response:
        level = level_for_mass(vol, mass) if mass >= 0 else level_for_sigma(vol, sigma)
        return Response(content=mesh_at(vol, level, smooth=max(0, min(50, smooth))),
                        media_type="application/octet-stream")

    @app.get("/api/grid")
    def grid(size: int = Query(96)) -> Response:
        """Density as a compact uint8 3D-texture payload (same layout as server.py):
        int32 nx, ny, nz, then float32 vmin, vmax, mean, std, then nx*ny*nz uint8."""
        ds = _downsample(vol, int(max(16, min(256, size))))
        vmin, vmax = float(ds.min()), float(ds.max())
        u8 = np.clip((ds - vmin) / (vmax - vmin + 1e-9) * 255.0, 0, 255).astype(np.uint8)
        nx, ny, nz = ds.shape[2], ds.shape[1], ds.shape[0]
        header = (
            np.array([nx, ny, nz], dtype="<i4").tobytes()
            + np.array([vmin, vmax, stats["mean"], stats["std"]], dtype="<f4").tobytes()
        )
        return Response(content=header + np.ascontiguousarray(u8).tobytes(),
                        media_type="application/octet-stream")

    @app.get("/api/mrc")
    def mrc() -> FileResponse:
        p = Path(filename)
        if not p.exists():
            raise HTTPException(status_code=404, detail="source .mrc not found")
        return FileResponse(str(p), media_type="application/octet-stream",
                            filename=p.name)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC / "mrc.html"))

    if STATIC.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC), html=False), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="WebGL map (.mrc) viewer backend")
    parser.add_argument("--mrc", required=True, help="path to a .mrc/.vol volume")
    parser.add_argument("--title", default="Map")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8052)
    args = parser.parse_args()

    volume = _load_mrc(args.mrc)
    app = build_app(volume, title=args.title, filename=str(Path(args.mrc).expanduser().resolve()))

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
