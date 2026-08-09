"""Standalone backend for the WebGL *map* viewer — one or many ``.mrc`` volumes.

This is the volume-viewer half of the conformational-landscape app, extracted so
any map a job produced (during or after a run) can be inspected on its own — no
network, no latents, no landscape. It loads each ``.mrc`` into a NumPy grid and
serves it as an isosurface mesh (marching cubes), a 3D-texture payload (for the
ray-marcher) and orthogonal slices, reusing the same helpers and frontend
renderers as ``server.py``:

    python mrc_server.py --mrc /path/a.mrc /path/b.mrc --port 8052

Several maps share one scene, one camera and one model panel — the ChimeraX
arrangement — so they can be compared, toggled, and morphed between. Every map is
placed in a *common* normalized cube (see ``_Scene.norm``), so boxes of different
sizes come out concentric and to relative scale rather than each filling the cube
and silently claiming to be the same size.

It is also spawned per-session by the main GUI backend, which reverse-proxies it
at ``/mrcviewer/<id>/`` (the frontend uses relative URLs, so it works under any
prefix).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

# Be a good GPU citizen even though this backend never touches the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Run as a plain script (the GUI spawns `python mrc_server.py`); the script's own
# directory is on sys.path, so the sibling `meshing.py` imports as a top-level module.

from fastapi import FastAPI, HTTPException, Query, Response  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from meshing import (_downsample, grid_box, level_for_mass, level_for_sigma, mesh_at,  # noqa: E402
                     painted_mesh_at, remove_dust)
import wirefmt  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

# Distinct, roughly equal-lightness hues for maps after the first. The first map keeps
# the viewer's neutral grey, as a lone map always has — colour should mean "there is
# more than one of these", not "this is a map".
PALETTE = ["#c8c8c8", "#7fb3ff", "#ff9e7a", "#8fe08f", "#e79aff", "#ffd166", "#6fdccf", "#ff8fa8"]


def _read_scale(path: str) -> tuple[float, list[float]]:
    """(voxel size in A, origin in A) from the MRC/CCP4 header, best-effort.

    These used to be discarded outright, which is what made the viewer purely
    qualitative: no A scale, and multi-map comparison assumed identical
    sampling. 1.0 / (0,0,0) when the header can't be read (e.g. Spider .vol)."""
    try:
        import mrcfile
        with mrcfile.open(path, mode="r", permissive=True, header_only=True) as m:
            vs = float(m.voxel_size.x) or 1.0
            o = m.header.origin
            origin = [float(o.x), float(o.y), float(o.z)]
            return (round(vs, 5), origin)
    except Exception:  # noqa: BLE001 - unsupported container: fall back to unscaled
        return (1.0, [0.0, 0.0, 0.0])


# Above this size the volume is memory-mapped read-only instead of loaded into
# RAM: a 512³ float32 map is 512 MB, and several open viewers used to stack those
# up. Everything downstream (stats, histogram, strided drafts, marching cubes,
# dust labelling, morph blends) only READS the base array — the operations that
# need writable data (blends, dust) allocate their own copies. The mrcfile
# handles are kept alive for the process lifetime; the viewer is per-session.
_MMAP_BYTES = int(os.environ.get("HAX_VIEWER_MMAP_MB", "512")) * 1024 * 1024
_OPEN_MMAPS: list[Any] = []


def _load_mrc(path: str) -> np.ndarray:
    """Read a volume into a writable float32 3D grid.

    `.mrc`/`.vol` go through xmipp's ImageHandler like everywhere else in hax.
    `.map`/`.ccp4` (and gzipped variants) are the *same* on-disk format, but
    ImageHandler does not dispatch those suffixes (BINARIES stays None and
    getData() returns None) — and `.map` is the extension every EMDB deposition
    ships under, so anyone comparing against a reference map hit a dead end.
    Read them with mrcfile directly: read-only + permissive, which also avoids
    ImageHandler's habit of rewriting the header of a file it fails to parse
    (deposited maps often live on read-only mounts)."""
    name = Path(path).name.lower()
    is_mrc_family = name.endswith((".mrc", ".map", ".ccp4"))
    # Big MRC-convention file: map it instead of reading it. (Gzipped variants
    # can't be mapped; ImageHandler formats have their own reader below.)
    if is_mrc_family and os.path.getsize(path) > _MMAP_BYTES:
        import mrcfile

        m = mrcfile.mmap(path, mode="r", permissive=True)
        if m.data is None or m.data.ndim not in (3, 4):
            m.close()
            raise ValueError(f"{path}: could not read volume data")
        _OPEN_MMAPS.append(m)                  # the array dies with the handle
        vol = m.data[0] if m.data.ndim == 4 else m.data
        if vol.dtype != np.float32:
            vol = np.asarray(vol, dtype=np.float32)   # non-float map: one real copy
        return vol
    if name.endswith((".map", ".ccp4", ".map.gz", ".mrc.gz", ".ccp4.gz")):
        import mrcfile

        with mrcfile.open(path, mode="r", permissive=True) as m:
            if m.data is None:
                raise ValueError(f"{path}: could not read volume data")
            vol = np.asarray(m.data, dtype=np.float32)
    else:
        from xmipp_metadata.image_handler import ImageHandler

        data = ImageHandler(path).getData()
        if data is None:
            raise ValueError(f"{path}: unsupported volume format")
        vol = np.asarray(data, dtype=np.float32)
    vol = np.squeeze(vol)
    if vol.ndim == 4:
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"{path} is not a 3D volume (got shape {vol.shape})")
    # Both readers can hand back a read-only view (mrcfile flags its data
    # read-only; ImageHandler can return an mmap view), and mesh_at / slicing
    # need a writable buffer. ascontiguousarray alone does NOT guarantee a
    # copy, so force one when needed.
    vol = np.ascontiguousarray(vol, dtype=np.float32)
    return vol if vol.flags.writeable else vol.copy()


class _Model:
    """One loaded map, plus the cached strided copies drafts are cut from."""

    def __init__(self, index: int, path: Path, vol: np.ndarray) -> None:
        self.index = index
        self.path = path
        self.name = path.name
        self.vol = vol
        self.voxel_size, self.origin = _read_scale(str(path))
        self._hist: dict[int, dict] = {}   # per-bins histogram cache (the volume never changes)
        self.stats = {
            "min": float(vol.min()), "max": float(vol.max()),
            "mean": float(vol.mean()), "std": float(vol.std()),
        }
        self.drafts: dict[int, tuple[np.ndarray, int]] = {}

    def grid_at(self, size: int) -> tuple[np.ndarray, int]:
        """The map itself, or a strided copy small enough for a live draft."""
        full = max(self.vol.shape)
        if not (0 < size < full):
            return self.vol, 1
        if size not in self.drafts:
            self.drafts[size] = _downsample(self.vol, int(size))
        return self.drafts[size]


class _Scene:
    """The loaded maps and the common frame they are all drawn in."""

    def __init__(self, models: list[_Model]) -> None:
        self.models = models
        # One normalization for the whole scene: the largest box edge anywhere. Every map
        # is then centred inside it, so two maps of the same size land exactly on top of
        # each other and a smaller one sits concentric at its true relative scale.
        self.full = max(max(m.vol.shape) for m in models)

    def norm(self, m: _Model) -> tuple[int, list[float]]:
        """(full_size, per-axis voxel offset) that places `m` in the common cube."""
        return self.full, [(self.full - n) / 2.0 for n in m.vol.shape]

    def get(self, index: int) -> _Model:
        if not 0 <= index < len(self.models):
            raise HTTPException(status_code=404, detail=f"no model {index}")
        return self.models[index]

    def morph_shape(self) -> tuple[int, ...] | None:
        """The shape every map shares, or None if they differ.

        Interpolating densities voxel-by-voxel is only meaningful on a common grid, so
        this is what decides whether the frontend may offer a morph at all. Resampling
        mismatched maps is deliberately not attempted here: it needs the pixel size and
        origin from the file headers to be right, and getting it silently wrong would
        produce a plausible-looking morph between misaligned maps.
        """
        shapes = {m.vol.shape for m in self.models}
        return self.models[0].vol.shape if len(shapes) == 1 else None


def build_app(scene: _Scene, title: str = "Map") -> FastAPI:
    app = FastAPI(title="Hax WebGL map viewer")

    # The frontend is plain ES modules served statically; never cache them so code
    # changes take effect on reload (mirrors server.py).
    @app.middleware("http")
    async def _no_cache(request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "models": len(scene.models)}

    @app.get("/api/info")
    def info() -> dict:
        return {
            "title": title,
            "full_size": scene.full,
            # Same grid everywhere => the frontend may offer to morph. None => it must not.
            "morph_shape": list(scene.morph_shape()) if scene.morph_shape() else None,
            "models": [
                {
                    "index": m.index,
                    "name": m.name,
                    "filename": str(m.path),
                    "shape": [int(s) for s in m.vol.shape],   # (nz, ny, nx), NumPy order
                    "voxel_size": m.voxel_size,               # A per voxel (1.0 = unknown)
                    "origin": m.origin,                       # A, from the header
                    "color": PALETTE[m.index % len(PALETTE)],
                    **m.stats,
                    "sigma_level": level_for_sigma(m.vol, 2.5),
                }
                for m in scene.models
            ],
        }

    @app.get("/api/histogram")
    def histogram(m: int = Query(0), bins: int = Query(96)) -> dict:
        model = scene.get(m)
        nb = int(max(8, min(256, bins)))
        cached = model._hist.get(nb)
        if cached is None:
            # A full-volume histogram is a whole-array pass (seconds at 512^3);
            # the volume never changes, so pay it once per bin count.
            counts, edges = np.histogram(model.vol.ravel(), bins=nb)
            cached = model._hist[nb] = {
                "counts": counts.astype(int).tolist(),
                "edges": edges.astype(float).tolist(),
                "sigma_level": level_for_sigma(model.vol, 2.5),
            }
        return {**cached, **model.stats}

    @app.get("/api/decode")
    def decode(
        m: int = Query(0),
        sigma: float = Query(2.5),
        mass: float = Query(-1.0),     # >=0 -> contour at constant enclosed mass
        smooth: int = Query(0),        # Taubin smoothing iterations on the surface
        size: int = Query(0),          # >0 -> draft mesh from a strided grid this big
        dust: int = Query(0),          # >0 -> hide components smaller than this (full-grid voxels)
    ) -> Response:
        model = scene.get(m)
        grid, step = model.grid_at(int(size))
        level = (level_for_mass(grid, mass / float(step) ** 3) if mass >= 0
                 else level_for_sigma(grid, sigma))
        if dust > 0:
            # The threshold is in full-resolution voxels; a strided draft's
            # components hold step³ fewer of them.
            grid = remove_dust(grid, level, max(1, int(dust) // step ** 3))
        full, off = scene.norm(model)
        return Response(content=mesh_at(grid, level, smooth=max(0, min(50, smooth)),
                                        step=step, full_size=full, offset=off),
                        media_type="application/octet-stream")

    @app.get("/api/paint")
    def paint(
        m: int = Query(...),           # the map whose surface is contoured
        other: int = Query(...),       # the map subtracted from it
        sigma: float = Query(2.5),
        smooth: int = Query(0),
        size: int = Query(0),
        prange: float = Query(-1.0),   # ± range of the ramp; <=0 -> 3× diff std
    ) -> Response:
        """Surface of map `m`, coloured by the local value of (m − other) — ChimeraX's
        `color sample` applied to a difference map. Red = density lost, near-white =
        unchanged, blue = gained. Same-grid only, like every voxel-wise op here."""
        ma, mb = scene.get(m), scene.get(other)
        if ma.vol.shape != mb.vol.shape:
            raise HTTPException(status_code=400,
                                detail=f"maps are on different grids: {ma.vol.shape} vs {mb.vol.shape}")
        grid, step = ma.grid_at(int(size))
        gb, step_b = mb.grid_at(int(size))
        if step_b != step:             # same shapes stride identically; belt and braces
            raise HTTPException(status_code=500, detail="draft strides diverged")
        diff = grid - gb
        level = level_for_sigma(grid, sigma)
        full, off = scene.norm(ma)
        return Response(content=painted_mesh_at(grid, diff, level,
                                                smooth=max(0, min(50, smooth)), step=step,
                                                full_size=full, offset=off,
                                                prange=prange if prange > 0 else None),
                        media_type="application/octet-stream")

    @app.get("/api/grid")
    def grid(m: int = Query(0), size: int = Query(96)) -> Response:
        """Density as a compact uint8 3D-texture payload (same layout as server.py):
        int32 nx, ny, nz, then float32 vmin, vmax, mean, std, then float32 boxLo[3],
        boxHi[3] (the model-space cube the grid occupies), then nx*ny*nz uint8."""
        model = scene.get(m)
        ds, step = _downsample(model.vol, int(max(16, min(256, size))))
        full, off = scene.norm(model)
        lo, hi = grid_box(model.vol.shape, ds.shape, step, full_size=full, offset=off)
        return Response(content=_grid_payload(ds, model.stats, lo, hi),
                        media_type="application/octet-stream")

    @app.get("/api/morph/grid")
    def morph_grid(a: int = Query(...), b: int = Query(...), t: float = Query(0.5),
                   size: int = Query(96)) -> Response:
        """The same payload for a density-space blend of two maps.

        Only used to *check* a frame server-side; playback interpolates the two uploaded
        textures on the GPU instead, which is why the morph runs at screen rate.
        """
        va, vb = _morph_pair(scene, a, b)
        f = float(np.clip(t, 0.0, 1.0))
        blend = (1.0 - f) * va + f * vb
        ds, step = _downsample(blend, int(max(16, min(256, size))))
        full, off = scene.norm(scene.get(a))
        lo, hi = grid_box(blend.shape, ds.shape, step, full_size=full, offset=off)
        stats = {"mean": float(blend.mean()), "std": float(blend.std())}
        return Response(content=_grid_payload(ds, stats, lo, hi),
                        media_type="application/octet-stream")

    @app.get("/api/morph/decode")
    def morph_decode(a: int = Query(...), b: int = Query(...), t: float = Query(0.5),
                     sigma: float = Query(2.5), mass: float = Query(-1.0),
                     smooth: int = Query(0), size: int = Query(0)) -> Response:
        """Isosurface of the blend of two maps — one frame of a surface-mode morph.

        `mass >= 0` contours at constant enclosed mass instead of constant sigma, which
        is what keeps the surface from breathing as density moves between the two maps.
        """
        va, vb = _morph_pair(scene, a, b)
        f = float(np.clip(t, 0.0, 1.0))
        blend = (1.0 - f) * va + f * vb
        step = 1
        if 0 < size < max(blend.shape):
            blend, step = _downsample(blend, int(size))
        level = (level_for_mass(blend, mass / float(step) ** 3) if mass >= 0
                 else level_for_sigma(blend, sigma))
        full, off = scene.norm(scene.get(a))
        return Response(content=mesh_at(blend, level, smooth=max(0, min(50, smooth)),
                                        step=step, full_size=full, offset=off),
                        media_type="application/octet-stream")

    @app.get("/api/morph/export")
    def morph_export(seq: str = Query(...), per: int = Query(24)) -> FileResponse:
        """Every frame of a morph as real full-resolution ``.mrc`` files, zipped.

        For figures and analyses made outside the GUI: the in-app recording is a
        webm of the rendered view, whereas this is the density itself, one map per
        frame, on the maps' own grid with the voxel size stamped.
        """
        try:
            idx = [int(s) for s in seq.split(",") if s.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="seq must be comma-separated model indices")
        if len(idx) < 2:
            raise HTTPException(status_code=400, detail="a morph needs at least two maps")
        mods = [scene.get(i) for i in idx]
        if len({m.vol.shape for m in mods}) != 1:
            raise HTTPException(status_code=400, detail="cannot morph maps on different grids")
        per = int(max(2, min(120, per)))
        n = (len(mods) - 1) * per + 1
        if n * mods[0].vol.nbytes > 2 * 1024 ** 3:
            raise HTTPException(status_code=400,
                                detail=f"{n} frames of this box would exceed 2 GB — reduce frames per leg")

        import tempfile
        import zipfile

        import mrcfile

        pad = max(3, len(str(n - 1)))
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        frame = tempfile.NamedTemporaryFile(suffix=".mrc", delete=False)
        try:
            with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as z:
                for k in range(n):
                    leg = min(k // per, len(mods) - 2)
                    t = (k - leg * per) / per
                    blend = np.ascontiguousarray(
                        (1.0 - t) * mods[leg].vol + t * mods[leg + 1].vol, dtype=np.float32)
                    with mrcfile.new(frame.name, overwrite=True) as f:
                        f.set_data(blend)
                        f.voxel_size = mods[0].voxel_size
                    z.write(frame.name, arcname=f"morph_{k:0{pad}d}.mrc")
        finally:
            os.unlink(frame.name)
        return FileResponse(tmp.name, media_type="application/zip", filename="morph_frames.zip")

    @app.post("/api/compute")
    def compute(op: str = Query(...), a: int = Query(...), b: int = Query(...)) -> dict:
        """Map arithmetic: add a derived model (A−B / A+B / A×B) to the scene.

        Same-grid only, like the morph — blending mismatched grids voxel-wise would
        be silently wrong. The result is written to a real temp .mrc (voxel size
        stamped from A) so the download button works on it like on any other map.
        """
        ma, mb = scene.get(a), scene.get(b)
        if ma.vol.shape != mb.vol.shape:
            raise HTTPException(status_code=400,
                                detail=f"maps are on different grids: {ma.vol.shape} vs {mb.vol.shape}")
        stem = lambda m_: m_.name.rsplit(".", 1)[0]  # noqa: E731
        if op == "diffpm":
            # Two-sided difference: the (+) model is the density A gained over B, the
            # (−) model the density it lost (i.e. B − A). Each is an ordinary map
            # contoured at a positive level, which is exactly the green/red pair a
            # difference-map figure uses — the frontend colours them accordingly.
            plus = _add_derived(scene, ma.vol - mb.vol, f"{stem(ma)} − {stem(mb)} (+).mrc", ma)
            minus = _add_derived(scene, mb.vol - ma.vol, f"{stem(ma)} − {stem(mb)} (−).mrc", ma)
            return {"index": plus.index, "name": plus.name, "pair": [plus.index, minus.index]}
        ops = {"diff": (np.subtract, "−"), "sum": (np.add, "+"), "mult": (np.multiply, "×")}
        if op not in ops:
            raise HTTPException(status_code=400, detail=f"unknown op {op!r} (diff/diffpm/sum/mult)")
        fn, sym = ops[op]
        model = _add_derived(scene, fn(ma.vol, mb.vol), f"{stem(ma)} {sym} {stem(mb)}.mrc", ma)
        return {"index": model.index, "name": model.name}

    @app.get("/api/pdb")
    def pdb(path: str = Query(...)) -> dict:
        """Backbone trace of a PDB file: CA (protein) / P (nucleic) atoms per chain,
        in Å. The frontend places it over the maps using voxel size + origin."""
        p, chains = _read_backbone(path)
        return {"name": p.name,
                "chains": [{"id": cid, "xyz": [c for xyz in pts for c in xyz]}
                           for cid, pts in chains.items()]}

    @app.get("/api/fit")
    def fit(path: str = Query(...), m: int = Query(0)) -> dict:
        """How well the model sits in the map: the density (in σ units) sampled at
        each backbone atom. A cheap, honest sanity number — mean σ well above 0
        and most atoms above 1σ means the fit is at least plausible; it is not a
        cross-correlation refinement score and does not pretend to be."""
        model = scene.get(m)
        _, chains = _read_backbone(path)
        xyz = np.array([pt for pts in chains.values() for pt in pts], dtype=np.float64)
        vs = model.voxel_size or 1.0
        org = np.asarray(model.origin, dtype=np.float64)
        # Å -> voxel indices; the volume is (z, y, x) so the sample coordinates flip.
        vox = (xyz - org[None, :]) / vs
        from scipy.ndimage import map_coordinates
        vals = map_coordinates(model.vol, [vox[:, 2], vox[:, 1], vox[:, 0]],
                               order=1, mode="constant", cval=np.nan)
        inside = np.isfinite(vals)
        n_in = int(inside.sum())
        if n_in == 0:
            raise HTTPException(status_code=400,
                                detail="no atoms fall inside this map — wrong map, or the "
                                       "model is in a different coordinate frame")
        sig = (vals[inside] - model.stats["mean"]) / (model.stats["std"] or 1.0)
        return {
            "n_atoms": int(len(xyz)),
            "n_inside": n_in,
            "mean_sigma": round(float(sig.mean()), 2),
            "median_sigma": round(float(np.median(sig)), 2),
            "frac_above_1sigma": round(float((sig > 1.0).mean()), 3),
        }

    @app.get("/api/mrc")
    def mrc(m: int = Query(0)) -> FileResponse:
        model = scene.get(m)
        if not model.path.exists():
            raise HTTPException(status_code=404, detail="source .mrc not found")
        return FileResponse(str(model.path), media_type="application/octet-stream",
                            filename=model.path.name)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC / "mrc.html"))

    if STATIC.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC), html=False), name="static")
    return app


def _read_backbone(path: str) -> tuple[Path, dict[str, list[list[float]]]]:
    """CA/P atoms per chain from a PDB file, in Å — shared by the overlay and the
    fit score so the two always agree on which atoms they mean."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"not found: {path}")
    if p.stat().st_size > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="file too large for a PDB")
    chains: dict[str, list[list[float]]] = {}
    try:
        opener = __import__("gzip").open if p.name.endswith(".gz") else open
        with opener(p, "rt", errors="replace") as f:
            for line in f:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                atom = line[12:16].strip()
                if atom not in ("CA", "P"):
                    continue
                if line[16] not in (" ", "A"):     # altloc: keep the first conformer
                    continue
                chain = line[21].strip() or "?"
                chains.setdefault(chain, []).append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"could not parse {p.name}: {exc}")
    if not chains:
        raise HTTPException(status_code=400,
                            detail=f"no CA/P atoms in {p.name} — is it a PDB-format file?")
    return p, chains


def _add_derived(scene: _Scene, vol: np.ndarray, name: str, like: _Model) -> _Model:
    """Append a computed grid as a new model, backed by a real temp .mrc so the
    download button works on it like on any loaded map. Scale is inherited from
    the map it was derived from."""
    import tempfile

    import mrcfile

    vol = np.ascontiguousarray(vol, dtype=np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".mrc", delete=False)
    with mrcfile.new(tmp.name, overwrite=True) as f:
        f.set_data(vol)
        f.voxel_size = like.voxel_size
    model = _Model(len(scene.models), Path(tmp.name), vol)
    model.name = name                      # display name, not the temp path
    model.voxel_size, model.origin = like.voxel_size, list(like.origin)
    scene.models.append(model)
    return model


def _morph_pair(scene: _Scene, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """The two maps to blend, refusing the pairs that cannot be blended honestly."""
    va, vb = scene.get(a).vol, scene.get(b).vol
    if va.shape != vb.shape:
        raise HTTPException(
            status_code=400,
            detail=f"cannot morph maps on different grids: {va.shape} vs {vb.shape}")
    return va, vb


def _grid_payload(ds: np.ndarray, stats: dict[str, Any], lo, hi) -> bytes:
    """Pack a downsampled grid into the binary layout the ray-marcher parses."""
    vmin, vmax = float(ds.min()), float(ds.max())
    u8 = np.clip((ds - vmin) / (vmax - vmin + 1e-9) * 255.0, 0, 255).astype(np.uint8)
    nx, ny, nz = ds.shape[2], ds.shape[1], ds.shape[0]
    header = (
        wirefmt.header(wirefmt.GRID)
        + np.array([nx, ny, nz], dtype="<i4").tobytes()
        + np.array([vmin, vmax, stats["mean"], stats["std"]], dtype="<f4").tobytes()
        + np.array(list(lo) + list(hi), dtype="<f4").tobytes()
    )
    return header + np.ascontiguousarray(u8).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="WebGL map (.mrc) viewer backend")
    parser.add_argument("--mrc", required=True, nargs="+",
                        help="one or more .mrc/.vol volumes, shown together in one scene")
    parser.add_argument("--title", default="Map")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8052)
    args = parser.parse_args()

    models = [
        _Model(i, Path(p).expanduser().resolve(), _load_mrc(p))
        for i, p in enumerate(args.mrc)
    ]
    app = build_app(_Scene(models), title=args.title)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
