"""Standalone backend for the WebGL conformational-landscape viewer.

Runs in the hax conda env (it imports JAX + hax to decode volumes). Loads a
trained network and a latents ``.npy``, builds embeddings of the latent space
(PCA + UMAP) and clusters, and decodes a 3D map -> isosurface mesh on demand.
Serves its own static frontend so it is fully usable on its own:

    python viewer/server.py --latents z.npy --reload <ckpt_dir> --port 8051

It is also spawned per-session by the main GUI backend, which reverse-proxies it
at /viewer/<id>/ (the frontend uses relative URLs so it works under any prefix).
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

# Be a good GPU citizen: don't preallocate the whole device.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# How many volume decodes may run at once (see _resolve_volume).
_DECODE_SLOTS = threading.BoundedSemaphore(
    max(1, int(os.environ.get("HAX_VIEWER_DECODE_CONCURRENCY", "2")))
)

# Timing note: the split between network decode and meshing has never been measured.
# The estimates in the comments below ("~700 ms of the ~750") were taken at a 320³ box at
# *full* resolution, but `size=` shrinks the meshing ~15x while shrinking the network by
# exactly nothing (it always evaluates the model's own box — see `_decode_latent`), so at
# playback resolution the ratio could be anything. HAX_VIEWER_TIMING=1 (see vlog.py)
# prints the real per-frame breakdown; measure rather than inherit the assumption.

# Extracted pieces (script-style imports: these files sit beside this script, whose
# directory Python puts on sys.path when it is run directly or spawned by the manager).
from exporter import Exporter, _parse_seeds  # noqa: E402
from landscape import Landscape  # noqa: E402
from meshing import enclosed_mass, grid_box, level_for_mass, level_for_sigma, mesh_at  # noqa: E402
from vlog import _TIMING, log_exc as _log_exc, timing as _timing  # noqa: E402

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC = Path(__file__).resolve().parent / "static"


class SaveLayer(BaseModel):
    name: str
    indices: list[int]
    format: str = "xmd"


class SnapCurve(BaseModel):
    embed: str = "pca"                 # which coordinate table the curve was drawn in
    points: list[list[float]] = []     # the dense spline samples, in that space


class PathSpec(BaseModel):
    indices: list[int] = []            # the states a morph walks through, in order


class FilterSpec(BaseModel):
    thr: float | None = None           # keep z < thr; None restores the full set
    k: int = 10                        # neighbours the z-score is measured over


# --------------------------------------------------------------------------
# Landscape: embeddings (PCA/UMAP) + clustering of the latent space.
# --------------------------------------------------------------------------
def _model_box(network) -> int | None:
    """The decoded volume's grid edge, read off a loaded model, or None if it can't be.

    Every het decoder allocates its output as ``jnp.zeros((B, s, s, s))`` and keeps ``s``
    on the module: the volume decoder submodule holds it as ``volume_size`` (HetSIREN,
    ReconSIREN) or ``xsize`` (Zernike3Deep), and the top-level network mirrors it as
    ``xsize``. Probed most-direct-first — the submodule attribute is literally the shape
    used to allocate the grid, the top-level one is a mirror that could in principle drift.

    Returns None rather than guessing, so the caller keeps the old decode-to-learn-it path.
    """
    probes = (
        lambda: network.delta_volume_decoder.volume_size,
        lambda: network.volume_decoder.volume_size,
        lambda: network.delta_volume_decoder.xsize,
        lambda: network.xsize,
        lambda: network.volume_size,
    )
    for probe in probes:
        try:
            box = int(probe())
        except Exception:  # noqa: BLE001 - attribute absent, or not an int; try the next
            continue
        if box > 0:
            return box
    return None


# --------------------------------------------------------------------------
# Decoder: load network + latents, decode a latent -> density grid (lazy JAX).
# --------------------------------------------------------------------------
class Decoder:
    def __init__(self, latents_path: str, reload_path: str, reduced_path: str | None = None) -> None:
        self.latents = _load_latents(latents_path)   # full latents, used for decoding
        display = self.latents
        if reduced_path:
            reduced = _load_latents(reduced_path)
            if reduced.shape[0] == self.latents.shape[0]:
                display = reduced   # show the reduced space; still decode the full latents
            else:
                print(f"[viewer] z_space_reduced has {reduced.shape[0]} rows but z_space has "
                      f"{self.latents.shape[0]}; ignoring the reduced space.", flush=True)
        self.landscape = Landscape(display, decode=self.latents)
        self._reload = reload_path
        self._decode = None  # built lazily on first decode (defers the JAX import)
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        # Small LRU for interpolated morph frames, keyed on a quantised (a, b, t):
        # smooths the *first* pass of a morph (the GUI caches whole timelines
        # client-side, so the server rarely sees a frame twice after that). Kept
        # deliberately small — each entry is one full-resolution volume, so this
        # is the cap that bounds server RAM here.
        self._icache: dict[tuple, np.ndarray] = {}
        self._icache_order: list[tuple] = []
        self._icache_cap = 16
        # Total RAM the two caches above may hold. The count caps are the first limit; this
        # is the one that actually bounds memory once the box gets large (see _trim_caches).
        self._budget = int(os.environ.get("HAX_VIEWER_VOL_CACHE_MB", "1024")) * 1024 * 1024
        self._box: int | None = None        # the model's grid size, learned on first decode
        self._decodes = 0                   # counter, for the HAX_VIEWER_TIMING breakdown
        self.extra: list[np.ndarray] = []   # centroid latents, addressed as indices n..n+m-1

    @property
    def total(self) -> int:
        return self.landscape.n + len(self.extra)

    def set_centroids(self, centroids: np.ndarray) -> int:
        """Register centroid latents as decodable virtual states; returns the base index."""
        self.extra = [np.asarray(c, dtype=np.float32) for c in centroids]
        for i in list(self._cache):     # drop cached centroid volumes (indices may be reused)
            if i >= self.landscape.n:
                self._cache.pop(i, None)
        self._cache_order = [i for i in self._cache_order if i < self.landscape.n]
        # interp frames touching a centroid index are likewise stale
        n = self.landscape.n
        for k in list(self._icache):
            if k[0] >= n or k[1] >= n:
                self._icache.pop(k, None)
        self._icache_order = [k for k in self._icache_order if k[0] < n and k[1] < n]
        return self.landscape.n

    def _latent(self, index: int) -> np.ndarray:
        n = self.landscape.n
        if 0 <= index < n:
            # Through the landscape, not `self.latents`: under a filter the two differ,
            # and the index the browser sent addresses the filtered set.
            return self.landscape.decode[index]
        if n <= index < n + len(self.extra):
            return self.extra[index - n]
        raise IndexError(f"latent index {index} out of range")

    def invalidate_states(self) -> None:
        """Drop everything keyed on a state index. Called when a filter renumbers them:
        cached volume 7 is no longer state 7, and a stale hit would decode the wrong map."""
        self.extra = []
        self._cache.clear()
        self._cache_order.clear()
        self._icache.clear()
        self._icache_order.clear()

    def latent(self, index: int) -> np.ndarray:
        """The decodable latent for a state index (centroids included)."""
        return self._latent(index)

    def _ensure_model(self) -> None:
        if self._decode is not None:
            return
        import jax
        import jax.numpy as jnp
        from flax import nnx

        from hax.checkpointer import NeuralNetworkCheckpointer

        network = NeuralNetworkCheckpointer.load(self._reload)
        graphdef, state = nnx.split(network)

        @jax.jit
        def _decode(x):
            model = nnx.merge(graphdef, state)
            return model.decode_volume(x)

        self._jnp = jnp
        self._decode = _decode
        # The box is a property of the checkpoint, so read it off the model here. `draft`
        # needs it to choose the stride, and it used to learn it by decoding a whole volume
        # *unstrided* and copying it home — ~85 ms of PCIe on top of the decode (see
        # `_decode_latent`), paid on the first morph frame of every session, purely to find
        # out an integer the model was already carrying. `_decode_latent` still sets the
        # authoritative value from the real output shape; this only avoids the trip.
        if self._box is None:
            self._box = _model_box(network)

    def _decode_latent(self, latent: np.ndarray, stride: int = 1) -> np.ndarray:
        """Decode one latent to a density grid. With ``stride > 1`` the grid is subsampled
        **on the device, before the copy home**.

        That ordering is the whole point. The network is cheap and barely cares about the
        box (it evaluates a fixed set of Gaussians and splats them), but the volume it
        splats into is not: at a 320³ box the result is 131 MB, and getting it across PCIe
        costs ~85 ms — measured *more* than meshing a draft of it. Striding on the host
        would pay that 85 ms and then throw 7/8 of the data away. Striding here copies
        only what we are going to mesh (~16 MB, ~10 ms), which is what makes a morph — a
        fresh latent every frame, so nothing is ever cached — playable at all.
        """
        load0 = time.perf_counter()
        self._ensure_model()
        load = time.perf_counter() - load0        # nonzero only on the very first call
        jnp = self._jnp
        t0 = time.perf_counter()
        vol = self._decode(jnp.asarray(latent))
        vol = jnp.squeeze(vol)
        if vol.ndim == 4:
            vol = vol[0]
        if vol.ndim != 3:
            raise RuntimeError(f"decode_volume returned shape {vol.shape}, expected a 3D grid")
        box = int(max(vol.shape))
        if self._box is not None and self._box != box:
            # _ensure_model read the box off the checkpoint; the real output disagrees. The
            # decoded shape wins, but say so — one frame was strided for the wrong box
            # before we got here, and _model_box's probe list needs updating for this model.
            print(f"[viewer] checkpoint reports box {self._box} but decode_volume returned "
                  f"{box}; using {box}.", flush=True)
        self._box = box                     # authoritative: the box is a property of the model
        if stride > 1:
            vol = vol[::stride, ::stride, ::stride]      # still a device op
        t1 = time.perf_counter()            # dispatch only so far: JAX ops are async
        if _TIMING:
            vol.block_until_ready()         # so `device` below is compute, not queueing
        t2 = time.perf_counter()
        # np.array (not asarray) forces a writable copy -- JAX arrays are
        # read-only and skimage.marching_cubes needs a writable buffer.
        out = np.ascontiguousarray(np.array(vol, dtype=np.float32))
        t3 = time.perf_counter()
        self._decodes += 1
        _timing(f"decode #{self._decodes}", dispatch=t1 - t0, device=t2 - t1, copy_home=t3 - t2,
                **({"model_load": load} if load > 0.01 else {}))
        if _TIMING and self._decodes == 1:
            print(f"[viewer:timing]   ^ first decode: includes the XLA trace/compile and the "
                  f"checkpoint load; box={box}, stride={stride}, out={out.shape}", flush=True)
        return out

    @property
    def box(self) -> int | None:
        """The model's grid size, or None until something has been decoded."""
        return self._box

    def has(self, index: int) -> bool:
        return index in self._cache

    def draft(self, index: int, index2: int | None, t: float, size: int) -> tuple[np.ndarray, int]:
        """A reduced-resolution grid to mesh, and the stride it was taken at.

        Two paths, and the split is what matters:

        * **a plain state** — its full volume is very likely already cached (it was decoded
          to show the map), so stride the cached array. No GPU work at all.
        * **a morph frame** — every ``t`` is a new latent, so nothing is or ever will be
          cached and a decode is unavoidable. Take the stride on the device (see
          ``_decode_latent``) so the full volume never crosses PCIe.
        """
        if self._box is None:
            self._ensure_model()                   # reads the box straight off the checkpoint
        if self._box is None:
            self.volume(index)                     # model didn't expose it: learn it the old way
        step = max(1, int(np.ceil(self._box / float(max(1, size)))))
        if step <= 1:                              # asked for a draft no coarser than the box
            vol = (self.volume_interp(index, index2, t) if index2 is not None
                   else self.volume(index))
            return vol, 1
        if index2 is None:
            if index in self._cache:
                cached = self._cache[index]
                return np.ascontiguousarray(cached[::step, ::step, ::step], dtype=np.float32), step
            return self._decode_latent(self._latent(index), stride=step), step
        tt = max(0.0, min(1.0, float(t)))
        latent = (1.0 - tt) * self._latent(index) + tt * self._latent(index2)
        return self._decode_latent(latent, stride=step), step

    def _cached_bytes(self) -> int:
        return (sum(v.nbytes for v in self._cache.values())
                + sum(v.nbytes for v in self._icache.values()))

    def _trim_caches(self) -> None:
        """Hold both volume caches inside a *byte* budget.

        The counts alone (8 states + 16 morph frames) are a memory cap only if you know
        how big a volume is. At a 320³ box each one is 131 MB, so the caps quietly
        authorised ~3.1 GB of resident volumes. Evict morph frames first: a morph re-fills
        its timeline cheaply, whereas the cached states are what the user is looking at.
        """
        while self._cached_bytes() > self._budget and len(self._icache_order) > 2:
            self._icache.pop(self._icache_order.pop(0), None)
        while self._cached_bytes() > self._budget and len(self._cache_order) > 1:
            self._cache.pop(self._cache_order.pop(0), None)

    def volume(self, index: int) -> np.ndarray:
        if index in self._cache:
            return self._cache[index]
        vol = self._decode_latent(self._latent(index))
        self._cache[index] = vol
        self._cache_order.append(index)
        while len(self._cache_order) > 8:  # keep at most the 8 most recent volumes...
            self._cache.pop(self._cache_order.pop(0), None)
        self._trim_caches()                # ...and fewer still if they are large
        return vol

    def volume_interp(self, a: int, b: int, t: float) -> np.ndarray:
        """Decode the latent interpolated between states a and b (for morphs).

        Cached on a quantised (a, b, t) key so a looping / there-and-back morph
        replays from the cache instead of re-running the JAX decode every frame."""
        key = (a, b, round(float(t), 3))
        hit = self._icache.get(key)
        if hit is not None:
            return hit
        latent = (1.0 - t) * self._latent(a) + t * self._latent(b)
        vol = self._decode_latent(latent)
        self._icache[key] = vol
        self._icache_order.append(key)
        while len(self._icache_order) > self._icache_cap:
            self._icache.pop(self._icache_order.pop(0), None)
        self._trim_caches()
        return vol


def _load_latents(path: str) -> np.ndarray:
    p = Path(path)
    arr = np.loadtxt(p) if p.suffix == ".txt" else np.load(p)
    return np.atleast_2d(np.asarray(arr, dtype=np.float32))


# --------------------------------------------------------------------------
# Mesh extraction (server-side marching cubes).
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def build_app(decoder: Decoder, title: str = "Conformational landscape",
              metadata: str | None = None, project_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Hax WebGL viewer")
    ls = decoder.landscape
    exporter = Exporter(decoder, metadata, project_dir)

    # The frontend is plain ES modules served statically; tell the browser never to
    # cache them so code changes always take effect on a reload (no stale app.js).
    @app.middleware("http")
    async def _no_cache(request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "n": int(ls.n), "dims": int(ls.dim)}

    # The landscape arrays go out as raw little-endian buffers, not JSON.
    #
    # This was by far the worst thing the backend did. `pca.tolist()` plus
    # `latents.tolist()` at a million states is ~19 million boxed Python floats, which
    # FastAPI then walks twice (response-model coercion, then the JSON encoder) and
    # serialises into a few hundred MB of text — minutes of work with the GIL held, so the
    # viewer's own HTTP server went deaf while it happened, and then the browser had to
    # JSON.parse the result. The same numbers as float32 are ~8x smaller and parse in no
    # time, because there is nothing to parse.
    #
    # Layout: int32 rows, int32 cols, then rows*cols float32 (C order).
    def _f32_matrix(a: np.ndarray) -> Response:
        a = np.ascontiguousarray(a, dtype=np.float32)
        head = np.array(a.shape[:2], dtype="<i4").tobytes()
        return Response(content=head + a.tobytes(), media_type="application/octet-stream")

    # Layout: int32 n, then n int32.
    def _i32_vector(a: np.ndarray) -> Response:
        a = np.ascontiguousarray(a, dtype="<i4")
        head = np.array([a.shape[0]], dtype="<i4").tobytes()
        return Response(content=head + a.tobytes(), media_type="application/octet-stream")

    def _out_of_range(index: int) -> HTTPException:
        """Reject a state index the landscape no longer has — and say so in the log.

        This used to be a bare 404. It is the one error a client can provoke by holding an
        index across a change to the state set (applying a filter renumbers them), and it
        was the only failure in this backend that produced a message in the browser and
        *nothing at all* on disk — which is precisely the case where a log is wanted. The
        counts are here because the useful question is always "out of range of what".
        """
        msg = (f"state {index} is out of range: the landscape has {ls.n} states"
               + (f" ({ls.n_all} before filtering)" if ls.filtered else "")
               + f" plus {len(decoder.extra)} generated")
        print(f"[viewer] {msg}", flush=True)
        return HTTPException(status_code=404, detail=msg)

    # Layout: int32 n, then n float32.
    def _f32_vector(a: np.ndarray) -> Response:
        a = np.ascontiguousarray(a, dtype="<f4")
        head = np.array([a.shape[0]], dtype="<i4").tobytes()
        return Response(content=head + a.tobytes(), media_type="application/octet-stream")

    @app.get("/api/landscape")
    def landscape() -> dict:
        """Just the metadata. The arrays are fetched separately, as binary."""
        return {
            "title": title,
            "n": int(ls.n),
            "n_all": int(ls.n_all),
            "filtered": bool(ls.filtered),
            "dims": int(ls.dim),
            "pca_var": ls.pca_var,
            "pca_k": int(ls.pca.shape[1]),
        }

    @app.post("/api/filter")
    def set_filter(body: FilterSpec) -> dict:
        """Drop the outlying states and refit the landscape on what is left.

        `thr: null` restores the full set. Either way the embeddings, clustering and
        z-scores are discarded and recomputed on the new set — the refit is the point, not
        a side effect — so the response is deliberately returned only once that is safe to
        ask for, and the browser reloads its arrays after it."""
        before = ls.n
        try:
            if body.thr is None:
                ls.set_filter(None)
            else:
                ls.filter_by_zscore(body.thr, body.k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        decoder.invalidate_states()   # index i means a different state now
        print(f"[viewer] filter: {before} -> {ls.n} of {ls.n_all} states"
              + (f" (z < {body.thr}, k = {body.k})" if body.thr is not None else " (reset)"), flush=True)
        # `removed` is how many are gone *now*, not the delta — a reset would otherwise
        # report a negative count of removed states.
        return {"n": int(ls.n), "n_all": int(ls.n_all), "removed": int(ls.n_all - ls.n),
                "filtered": bool(ls.filtered)}

    @app.get("/api/pca.bin")
    def pca_bin() -> Response:
        return _f32_matrix(ls.pca)

    @app.get("/api/latents.bin")
    def latents_bin() -> Response:
        # Only the "latent" embedding view needs these, so they are no longer part of the
        # first page load — which used to ship the entire latent space to the browser
        # before it had drawn a single point.
        return _f32_matrix(ls.latents)

    @app.get("/api/umap.bin")
    def umap_bin() -> Response:
        return _f32_matrix(ls.umap())

    @app.get("/api/cluster.bin")
    def cluster_bin(k: int = Query(5)) -> Response:
        return _i32_vector(ls.clusters(k))

    @app.get("/api/rows.bin")
    def rows_bin() -> Response:
        """Which original particle each current state is. Identity until a filter is
        applied; after one it is what lets the browser carry its layers and its morph
        path across the renumbering instead of losing them."""
        return _i32_vector(ls.rows)

    @app.get("/api/zscore.bin")
    def zscore_bin(k: int = Query(10)) -> Response:
        """Per-state outlier z-scores (see Landscape.zscores).

        One request per k, then the browser thresholds the vector itself — which is what
        makes the threshold a slider rather than a job."""
        try:
            return _f32_vector(ls.zscores(k))
        except Exception as exc:  # noqa: BLE001
            _log_exc("z-score computation failed")
            raise HTTPException(status_code=500, detail=f"neighbour search failed: {exc}")

    @app.get("/api/kmeans")
    def kmeans(k: int = Query(5), seeds: str = Query("")) -> dict:
        """KMeans -> per-state labels + decodable centroids (virtual indices base..base+k-1).
        ``seeds`` is an optional comma-separated list of state indices used as the
        initial cluster centers (then k = number of unique seeds)."""
        seed_list = _parse_seeds(seeds)
        _labels, centroids, cpca = ls.kmeans_full(k, seed_list)
        base = decoder.set_centroids(centroids)
        # The per-state labels come back through /api/kmeans.bin instead of riding along
        # here as a million boxed Python ints. kmeans_full is memoised, so that second
        # call is a dict lookup, not a second clustering.
        return {
            "k": int(centroids.shape[0]),
            "base": int(base),
            "centroid_pca": cpca.tolist(),
        }

    @app.post("/api/snap")
    def snap(body: SnapCurve) -> dict:
        """Snap a drawn curve onto the states it passes through (see Landscape.snap)."""
        if not body.points:
            return {"indices": []}
        try:
            return {"indices": ls.snap(body.embed, body.points)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/kmeans.bin")
    def kmeans_bin(k: int = Query(5), seeds: str = Query("")) -> Response:
        labels, _c, _p = ls.kmeans_full(k, _parse_seeds(seeds))
        return _i32_vector(labels)

    @app.get("/api/centroid_coords")
    def centroid_coords(embed: str = Query("pca"), k: int = Query(5), seeds: str = Query("")) -> dict:
        """Centroid positions in one embedding's coordinates (see Landscape.centroid_coords).

        The browser asks for these when it switches to an embedding it hasn't got centroid
        coordinates for yet — a marker drawn in PCA units on a UMAP frame lands nowhere
        near its cluster."""
        embed = embed if embed in ("pca", "umap", "latent") else "pca"
        return {"embed": embed, "coords": ls.centroid_coords(embed, k, _parse_seeds(seeds)).tolist()}

    def _resolve_volume(index: int, index2: int | None, t: float) -> np.ndarray:
        """The decoded volume. Cheap on a repeat: `Decoder.volume` caches, so moving the
        contour slider never re-runs the network — only the mesh below is recomputed.

        The caller must hold `_DECODE_SLOTS`."""
        if index < 0 or index >= decoder.total:
            raise _out_of_range(index)
        if index2 is not None and 0 <= index2 < decoder.total and index2 != index:
            return decoder.volume_interp(index, index2, max(0.0, min(1.0, t)))
        return decoder.volume(index)

    # Strided copies of a *state's* volume, cached per (state, size). A contour drag
    # re-meshes the same volume over and over; without this, every frame would pay the
    # stride-copy — a 131 MB strided read at a 320³ box, the same order as the draft mesh
    # itself. Morph frames are deliberately NOT cached here: every `t` is a distinct
    # latent, so nothing would ever be reused, and the client caches the finished meshes.
    _drafts: dict[tuple, tuple[np.ndarray, int]] = {}
    _draft_order: list[tuple] = []

    def _draft_grid(index: int, index2: int | None, t: float, size: int) -> tuple[np.ndarray, int]:
        if index2 is not None:
            return decoder.draft(index, index2, t, size)      # morph frame: decode strided
        key = (index, int(size))
        hit = _drafts.get(key)
        if hit is not None:
            return hit
        ds, step = decoder.draft(index, None, 0.0, size)
        _drafts[key] = (ds, step)
        _draft_order.append(key)
        while len(_draft_order) > 4:               # ~16 MB each at 160³ — bound it
            _drafts.pop(_draft_order.pop(0), None)
        return ds, step

    @app.get("/api/decode")
    def decode(
        index: int = Query(...),
        sigma: float = Query(2.5),
        mass: float = Query(-1.0),       # >=0 -> contour at constant enclosed mass
        index2: int = Query(-1),         # >=0 -> interpolate index..index2 at t
        t: float = Query(0.0),
        smooth: int = Query(0),          # Taubin smoothing iterations on the surface
        size: int = Query(0),            # >0 -> draft mesh from a strided grid this big
    ) -> Response:
        # The *whole* route runs under the slot, not just the decode. Marching cubes plus
        # Taubin smoothing is the expensive half (~700 ms of the ~750 at a 320³ box), and
        # a client cannot cancel it: this is a sync `def`, so Starlette runs it in a
        # threadpool task, and aborting the fetch frees only the browser. Leaving the mesh
        # outside the slot let an aborting client stack up abandoned meshes — and they do
        # not parallelise, they contend (8 at once measured 5x slower *each*), so a
        # threshold drag got slower the further it went.
        if index < 0 or index >= decoder.total:
            raise _out_of_range(index)
        # -1 is how the client asks for a single state, so it stays the one value that
        # means "no second endpoint". Any *other* out-of-range index2 is a stale reference,
        # and is now rejected rather than dropped: silently ignoring it turned a morph frame
        # into a still of its own start point, so the map stopped moving part-way through
        # and nothing — not the response, not the log — said why.
        if index2 < -1 or index2 >= decoder.total:
            raise _out_of_range(index2)
        other = index2 if (0 <= index2 < decoder.total and index2 != index) else None
        with _DECODE_SLOTS:
            t_grid = time.perf_counter()
            try:
                if size > 0:
                    # Draft: mesh a strided grid instead of the full one. ~15x cheaper at
                    # 320³ (a ~65 ms frame against ~1240 ms), which is what makes both a
                    # contour drag and a morph live. The level is taken from the strided
                    # grid too — millions of samples, so its mean/std match the full
                    # volume's to 4 decimals — keeping the whole request off the big array.
                    grid, step = _draft_grid(index, other, t, int(size))
                else:
                    grid, step = _resolve_volume(index, other, t), 1
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                _log_exc("decode failed")
                raise HTTPException(status_code=500, detail=f"decode failed: {exc}")

            t_level = time.perf_counter()
            full = decoder.box or max(grid.shape)
            level = (level_for_mass(grid, mass / float(step) ** 3) if mass >= 0
                     else level_for_sigma(grid, sigma))
            t_mesh = time.perf_counter()
            body = mesh_at(grid, level, smooth=max(0, min(50, smooth)),
                           step=step, full_size=full)
            t_end = time.perf_counter()
            # The number the whole performance question turns on: how a playback frame
            # actually divides between the network and the host-side mesher.
            _timing("/api/decode", grid=t_level - t_grid, level=t_mesh - t_level,
                    mesh=t_end - t_mesh, total=t_end - t_grid)
        return Response(content=body, media_type="application/octet-stream")

    @app.get("/api/grid")
    def grid(
        index: int = Query(...),
        index2: int = Query(-1),
        t: float = Query(0.0),
        size: int = Query(96),       # max grid edge of the streamed texture
    ) -> Response:
        """Decoded density as a compact uint8 3D-texture payload for ray-marching.

        Layout: int32 nx, ny, nz (texture width/height/depth, x fastest), then
        float32 vmin, vmax, mean, std, then float32 boxLo[3], boxHi[3] (the model-space
        cube the grid occupies, matching the surface mesh), then nx*ny*nz uint8 voxels
        (C order)."""
        if index < 0 or index >= decoder.total:
            raise _out_of_range(index)
        # -1 is how the client asks for a single state, so it stays the one value that
        # means "no second endpoint". Any *other* out-of-range index2 is a stale reference,
        # and is now rejected rather than dropped: silently ignoring it turned a morph frame
        # into a still of its own start point, so the map stopped moving part-way through
        # and nothing — not the response, not the log — said why.
        if index2 < -1 or index2 >= decoder.total:
            raise _out_of_range(index2)
        other = index2 if (0 <= index2 < decoder.total and index2 != index) else None
        # The texture is a subsample, so decode straight to it — striding on the device
        # rather than hauling the full volume home and throwing most of it away. This is
        # what a morph frame costs in the ray-marched render modes.
        with _DECODE_SLOTS:
            try:
                ds, step = _draft_grid(index, other, t, int(max(16, min(160, size))))
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                _log_exc("decode failed")
                raise HTTPException(status_code=500, detail=f"decode failed: {exc}")
        vmin, vmax = float(ds.min()), float(ds.max())
        u8 = np.clip((ds - vmin) / (vmax - vmin + 1e-9) * 255.0, 0, 255).astype(np.uint8)
        # numpy C order has the last axis varying fastest -> that is the texture x.
        nx, ny, nz = ds.shape[2], ds.shape[1], ds.shape[0]
        full = decoder.box or max(ds.shape) * step
        # The stats come from the strided grid: millions of samples, so its mean/std match
        # the full volume's — and taking them here keeps the full array out of the request.
        lo, hi = grid_box((full, full, full), ds.shape, step)
        header = (
            np.array([nx, ny, nz], dtype="<i4").tobytes()
            + np.array([vmin, vmax, float(ds.mean()), float(ds.std())], dtype="<f4").tobytes()
            + np.array(lo + hi, dtype="<f4").tobytes()
        )
        return Response(content=header + np.ascontiguousarray(u8).tobytes(), media_type="application/octet-stream")

    @app.post("/api/segments")
    def segments(body: PathSpec) -> dict:
        """The latent-space length of each consecutive hop in a path.

        A morph allocates its frames in proportion to these, so it advances at a constant
        speed through latent space instead of giving every hop the same frame count
        regardless of how far it travels.

        The distances have to be taken in the **full** latent space — the space the decoder
        actually interpolates in — not in the 2-D/3-D projection the landscape is drawn in.
        A projection foreshortens: two states that look adjacent on screen can be far apart
        in the space the morph traverses, and allocating frames from the picture would give
        that hop too few and make the map jump through it.
        """
        idx = [int(i) for i in body.indices if 0 <= int(i) < decoder.total]
        if len(idx) < 2:
            return {"lengths": []}
        lat = np.stack([decoder.latent(i) for i in idx]).astype(np.float64)
        d = np.linalg.norm(np.diff(lat, axis=0), axis=1)
        return {"lengths": [float(x) for x in d]}

    @app.get("/api/mass")
    def mass_at(index: int = Query(...), sigma: float = Query(2.5), size: int = Query(0)) -> dict:
        """The density a state encloses at a given sigma — the reference a constant-mass
        morph contours every frame to.

        Its own route because the client wants the *number*, not a picture. A morph is
        contoured at constant mass (a fixed sigma would make the map breathe as the latent
        moves), so dragging the contour during one has to re-derive this reference on every
        pointermove. Asking `/api/decode` for it would mesh the state each time — ~30x the
        cost, for a mesh thrown away on arrival. On a cached draft grid this is a mean, a
        std and one masked sum: a few ms.
        """
        if index < 0 or index >= decoder.total:
            raise _out_of_range(index)
        with _DECODE_SLOTS:
            try:
                grid, step = (_draft_grid(index, None, 0.0, int(size)) if size > 0
                              else (_resolve_volume(index, None, 0.0), 1))
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                _log_exc("decode failed")
                raise HTTPException(status_code=500, detail=f"decode failed: {exc}")
            level = level_for_sigma(grid, sigma)
            # Scale back up by the stride so a draft's mass is comparable to a full one —
            # the same convention `mesh_at` uses for the mass it reports in its header.
            return {"level": level, "mass": enclosed_mass(grid, level) * float(step) ** 3}

    @app.get("/api/histogram")
    def histogram(index: int = Query(...), bins: int = Query(96)) -> dict:
        if index < 0 or index >= decoder.total:
            raise _out_of_range(index)
        vol = decoder.volume(index)
        counts, edges = np.histogram(vol.ravel(), bins=int(max(8, min(256, bins))))
        return {
            "counts": counts.astype(int).tolist(),
            "edges": edges.astype(float).tolist(),
            "min": float(vol.min()), "max": float(vol.max()),
            "mean": float(vol.mean()), "std": float(vol.std()),
            "sigma_level": level_for_sigma(vol, 2.5),
        }

    @app.get("/api/mrc")
    def mrc(index: int = Query(...)) -> Response:
        if index < 0 or index >= decoder.total:
            raise _out_of_range(index)
        import tempfile
        from starlette.background import BackgroundTask
        from xmipp_metadata.image_handler import ImageHandler

        vol = decoder.volume(index)
        with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as tf:
            path = tf.name
        try:
            ImageHandler().write(vol, path, overwrite=True)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        # Stream the file instead of slurping it back into RAM (two extra full
        # copies of the map, per download); the temp file is removed after the
        # response has been sent.
        return FileResponse(
            path,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="state_{index:04d}.mrc"'},
            background=BackgroundTask(lambda: Path(path).unlink(missing_ok=True)),
        )

    @app.get("/api/export_info")
    def export_info() -> dict:
        return {
            "available": exporter.available,
            "has_metadata": exporter.metadata_path is not None,
            "project": str(exporter.project_dir) if exporter.project_dir else None,
        }

    @app.post("/api/save_layer")
    def save_layer(body: SaveLayer) -> dict:
        try:
            return exporter.save_layer(body.name, body.indices, body.format)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            _log_exc("save_layer failed")
            raise HTTPException(status_code=500, detail=f"save failed: {exc}")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC / "index.html"))

    if STATIC.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
    return app


def select_gpu(gpu: str | None) -> None:
    """Pin this process to one device, before JAX ever loads.

    Same contract as ``hax_project_manager --gpu``: an index picks that device;
    ``cpu`` (or an empty string) hides every device and falls back to the CPU.
    Passing None leaves the inherited environment alone, so JAX chooses for itself
    — what the viewer did before the device was selectable.

    Ordering is the whole point, and it is not fragile here: JAX only reads
    CUDA_VISIBLE_DEVICES when it first initialises its backend, and this module
    defers ``import jax`` all the way to the first decode request (see
    ``Decoder._ensure_model``). Anything set before uvicorn starts serving — this
    included — lands well ahead of it.
    """
    if gpu is None:
        return
    choice = gpu.strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if choice.lower() in ("", "cpu") else choice


def main() -> None:
    parser = argparse.ArgumentParser(description="WebGL conformational-landscape viewer backend")
    parser.add_argument("--latents", required=True, help="latents .npy/.txt of shape (N, latent_dim)")
    parser.add_argument("--reload", required=True, help="trained network checkpoint dir (e.g. <run>/HetSIREN)")
    parser.add_argument("--z_space_reduced", dest="reduced", default=None,
                        help="optional reduced-dimensionality latents (row-aligned) used only to display the "
                             "landscape; volumes are still decoded from --latents (e.g. a FlexConsensus consensus)")
    parser.add_argument("--title", default="Conformational landscape")
    parser.add_argument("--metadata", default=None, help="row-aligned particle metadata (.xmd/.star) for layer export")
    parser.add_argument("--project-dir", dest="project_dir", default=None, help="project folder where exported layers are written")
    parser.add_argument("--gpu", default=None,
                        help="device to load the network and decode maps on: a GPU index (e.g. 0), or 'cpu' for none. "
                             "Omit to inherit CUDA_VISIBLE_DEVICES and let JAX choose.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    args = parser.parse_args()

    select_gpu(args.gpu)   # must precede the first JAX import; see select_gpu()

    decoder = Decoder(args.latents, args.reload, reduced_path=args.reduced)
    app = build_app(decoder, title=args.title, metadata=args.metadata, project_dir=args.project_dir)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
