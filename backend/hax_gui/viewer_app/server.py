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

# HAX_VIEWER_TIMING=1 prints a per-frame breakdown of where a morph frame's time actually
# goes: network vs copy-home vs marching cubes + smoothing.
#
# That split has never been measured. The estimates in the comments below ("~700 ms of the
# ~750") were taken at a 320³ box at *full* resolution, but `size=` shrinks the meshing ~15x
# while shrinking the network by exactly nothing (it always evaluates the model's own box —
# see `_decode_latent`), so at playback resolution the ratio could be anything. Whether it is
# worth attacking the decode or the mesher depends entirely on this number, so measure it
# rather than inheriting the assumption.
_TIMING = os.environ.get("HAX_VIEWER_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}


def _timing(stage: str, **parts: float) -> None:
    """Print one timing line. `parts` are seconds; shown as ms in declaration order."""
    if not _TIMING:
        return
    body = " | ".join(f"{k} {v * 1e3:8.1f} ms" for k, v in parts.items())
    print(f"[viewer:timing] {stage:12s} {body}", flush=True)

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


# --------------------------------------------------------------------------
# Landscape: embeddings (PCA/UMAP) + clustering of the latent space.
# --------------------------------------------------------------------------
class Landscape:
    def __init__(self, latents: np.ndarray, decode: np.ndarray | None = None) -> None:
        # `latents` drives the display (embedding/clustering); `decode` is the
        # full latent space whose rows decode to volumes. They differ only when a
        # reduced display space (z_space_reduced) is supplied — then centroids are
        # returned in the decodable full space so generated states still work.
        self.latents = latents
        self.decode = decode if decode is not None else latents
        self.n, self.dim = latents.shape
        self._pca = None
        self._pca_var = None
        self._pca_comp = None     # (k, dim) basis, for projecting new points (centroids)
        self._pca_mean = None
        self._umap = None
        self._clusters: dict[int, list[int]] = {}
        self._kmeans: dict[tuple, tuple] = {}   # (k, seeds, space) -> (labels, centroids, cpca)
        self._trees: dict[str, object] = {}     # embed -> cKDTree, for snapping drawn curves

    @property
    def pca(self):
        if self._pca is None:
            k = int(min(8, self.dim, max(2, self.n - 1)))
            try:
                from sklearn.decomposition import PCA
                p = PCA(n_components=k)
                emb = p.fit_transform(self.latents)
                self._pca = emb.astype(np.float32)
                self._pca_var = p.explained_variance_ratio_.astype(float).tolist()
                self._pca_comp = p.components_.astype(np.float32)
                self._pca_mean = p.mean_.astype(np.float32)
            except Exception:  # noqa: BLE001 - fall back to a raw SVD
                self._pca_mean = self.latents.mean(0).astype(np.float32)
                x = self.latents - self._pca_mean
                _, s, vt = np.linalg.svd(x, full_matrices=False)
                self._pca_comp = vt[:k].astype(np.float32)
                self._pca = (x @ self._pca_comp.T).astype(np.float32)
                ev = (s[:k] ** 2)
                self._pca_var = (ev / ev.sum()).astype(float).tolist()
        return self._pca

    # Above this many states, Lloyd's algorithm with ten restarts is minutes of work, and
    # it was being run *before the first frame was drawn* (see the /api/landscape route).
    # MiniBatchKMeans converges on the same structure from samples, in seconds.
    MINIBATCH_N = 50_000

    def _fit_labels(self, data: np.ndarray, k: int, init=None) -> np.ndarray:
        """Cluster `data` into k groups. One place, so every caller gets the same policy."""
        n = data.shape[0]
        if n > self.MINIBATCH_N:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(
                n_clusters=k,
                init=init if init is not None else "k-means++",
                n_init=1 if init is not None else 3,
                batch_size=4096,
                max_iter=200,
                random_state=0,
            )
            return km.fit_predict(data)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, init=init if init is not None else "k-means++",
                    n_init=1 if init is not None else 10, random_state=0)
        return km.fit_predict(data)

    def kmeans_full(self, k: int, seeds: list[int] | None = None):
        """KMeans -> (labels, decodable centroid latents, centroid PCA coords).

        Clustering always runs on the display latents (the reduced space when one was
        supplied, else the full latents) — i.e. on latent coordinates, never on the
        PCA/UMAP projection, which is display-only. Each row maps to one state, so labels
        and centroids project onto whatever embedding is on screen, and the centroids are
        the per-cluster mean of the *decode* latents so they always remain decodable.

        When ``seeds`` (state indices) are given, they are used as the initial
        cluster centers and fix the cluster count (k = number of unique seeds)."""
        _ = self.pca  # ensure the PCA basis exists
        data = self.latents
        seeds = [int(s) for s in (seeds or []) if 0 <= int(s) < self.n]
        seeds = list(dict.fromkeys(seeds))  # de-dup, keep order
        # Memoised: this used to be the one clustering path with no cache at all, so every
        # click of "Cluster" re-ran a full KMeans over every state.
        ckey = (int(k), tuple(seeds))
        hit = self._kmeans.get(ckey)
        if hit is not None:
            return hit
        try:
            init = data[seeds] if len(seeds) >= 2 else None
            labels = self._fit_labels(data, len(seeds) if init is not None else int(max(2, min(20, k))),
                                      init=init).astype(int)
        except Exception:  # noqa: BLE001 - degenerate fallback: one cluster
            labels = np.zeros(self.n, dtype=int)
        uniq = np.unique(labels)
        # decodable centroids: mean of the full (decode) latents per cluster
        centroids = np.stack([self.decode[labels == c].mean(0) for c in uniq]).astype(np.float32)
        # display positions: mean of the display latents per cluster, projected to PCA
        disp_centers = np.stack([self.latents[labels == c].mean(0) for c in uniq]).astype(np.float32)
        cpca = ((disp_centers - self._pca_mean) @ self._pca_comp.T).astype(np.float32)
        out = (labels, centroids, cpca)
        self._kmeans[ckey] = out
        return out

    def centroid_coords(self, embed: str, k: int, seeds: list[int] | None = None) -> np.ndarray:
        """Where the cluster centroids sit *in the coordinate table `embed` is drawn from*.

        A centroid marker has to be plotted in the same units as the points around it.
        Only PCA is a linear map of the display latents, so a stored PCA vector cannot be
        reused for the UMAP or raw-latent views — UMAP in particular has no usable basis
        to project a new point through. Instead take, per cluster, the mean of its
        members' coordinates in that embedding: defined for every embedding, and it lands
        the marker in the middle of the cluster it belongs to, which is what it means.

        ``kmeans_full`` is memoised, so this reuses the existing labels."""
        labels, _c, _p = self.kmeans_full(k, seeds)
        coords = np.asarray(self._coords(embed))
        return np.stack([coords[labels == c].mean(0) for c in np.unique(labels)]).astype(np.float32)

    @property
    def pca_var(self):
        if self._pca_var is None:
            _ = self.pca
        return self._pca_var

    def umap(self) -> np.ndarray:
        # 3 components so the same embedding drives both the 2D and 3D views
        # (the 2D view just uses the first two columns), mirroring PCA.
        if self._umap is None:
            k = int(min(3, max(2, self.n - 1)))
            try:
                import umap
                reducer = umap.UMAP(n_components=k, random_state=42)
                self._umap = reducer.fit_transform(self.latents).astype(np.float32)
            except Exception:  # noqa: BLE001 - degrade to the first PCs
                self._umap = self.pca[:, :k].copy()
        return self._umap

    def _coords(self, embed: str) -> np.ndarray:
        """The coordinate table a given embedding view is drawn from."""
        if embed == "umap":
            return self.umap()
        if embed == "latent":
            return self.latents
        return self.pca

    def snap(self, embed: str, pts: np.ndarray) -> list[int]:
        """Nearest state to each point of a drawn curve → an ordered, deduped sequence.

        This is the query the browser used to answer itself, by scanning every state for
        each of the ~550 samples along the spline: ~4e9 distance evaluations, seconds of a
        frozen tab, on a plain click of ▶. A k-d tree answers the same thing exactly, in
        tens of milliseconds, and the tree is built once per embedding and kept.

        (A KMeans/IVF index over the same points is also exact, via the triangle
        inequality, but measurably slower to both build and query at these dimensions —
        eight dims is still comfortably inside what a k-d tree handles well.)
        """
        coords = np.ascontiguousarray(self._coords(embed), dtype=np.float32)
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float32))
        if pts.ndim != 2 or pts.shape[1] != coords.shape[1]:
            raise ValueError(f"curve points have {pts.shape[1]} dims, but the "
                             f"'{embed}' space has {coords.shape[1]}")
        tree = self._trees.get(embed)
        if tree is None:
            try:
                from scipy.spatial import cKDTree
                tree = self._trees[embed] = cKDTree(coords)
            except ImportError:
                tree = None
        if tree is not None:
            _dist, idx = tree.query(pts, k=1, workers=-1)
        else:
            # No scipy: an exact blocked brute force. Still far better than the browser
            # doing it, because BLAS is doing the work and it is off the UI thread.
            best_d = np.full(len(pts), np.inf, np.float64)
            idx = np.zeros(len(pts), np.int64)
            qsq = (pts ** 2).sum(1)[:, None]
            for s in range(0, coords.shape[0], 100_000):
                blk = coords[s:s + 100_000]
                d = qsq - 2.0 * (pts @ blk.T) + (blk ** 2).sum(1)[None, :]
                j = d.argmin(1)
                v = d[np.arange(len(pts)), j]
                m = v < best_d
                best_d[m] = v[m]
                idx[m] = j[m] + s

        seq: list[int] = []
        for i in np.atleast_1d(idx):
            i = int(i)
            if not seq or seq[-1] != i:   # consecutive samples land on the same state a lot
                seq.append(i)
        return seq

    def clusters(self, k: int) -> np.ndarray:
        k = int(max(1, min(20, k)))
        if k not in self._clusters:
            try:
                labels = self._fit_labels(self.latents, k)
            except Exception:  # noqa: BLE001
                labels = np.zeros(self.n, dtype=int)
            # int32, not a list of Python ints: this is shipped as binary now, and at a
            # million states the list alone was tens of MB of boxed objects.
            self._clusters[k] = np.asarray(labels, dtype=np.int32)
        return self._clusters[k]


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
            return self.latents[index]
        if n <= index < n + len(self.extra):
            return self.extra[index - n]
        raise IndexError(f"latent index {index} out of range")

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
def level_for_sigma(vol: np.ndarray, sigma: float) -> float:
    """Contour level at mean + sigma*std (scale-invariant cryo-EM convention)."""
    return float(vol.mean()) + float(sigma) * (float(vol.std()) or 1.0)


def enclosed_mass(vol: np.ndarray, level: float) -> float:
    """Total density above `level` -- the 'mass' kept constant during a morph."""
    return float(vol[vol >= level].sum())


def level_for_mass(vol: np.ndarray, target_mass: float, bins: int = 1 << 16) -> float:
    """The level whose enclosed mass equals `target_mass`. Used to morph at constant mass
    (a fair shape comparison), so it runs on *every* morph frame.

    Solved from one weighted histogram rather than by bisection. Mass above a level is a
    reverse cumulative sum over the histogram, which is monotone, so the answer is a
    lookup plus a linear interpolation inside the bracketing bin.

    The bisection this replaces evaluated `enclosed_mass` 40 times, and each of those
    built a full boolean mask and a fancy-indexed copy of the volume — at a 320³ box that
    is ~40 × 160 MB of pure memory traffic per frame, none of it releasing the GIL. This
    is a single pass instead.
    """
    flat = vol.ravel()
    lo, hi = float(flat.min()), float(flat.max())
    if not (hi > lo):
        return lo
    # mass (sum of voxel values), binned by voxel value
    per_bin, edges = np.histogram(flat, bins=bins, range=(lo, hi), weights=flat)
    # above[i] = mass of every voxel whose value is >= edges[i]; non-increasing in i
    above = np.empty(len(edges), dtype=np.float64)
    above[:-1] = np.cumsum(per_bin[::-1])[::-1]
    above[-1] = 0.0
    # first edge whose enclosed mass has dropped to or below the target
    i = int(np.searchsorted(-above, -float(target_mass), side="left"))
    if i <= 0:
        return lo
    if i >= len(edges):
        return hi
    m0, m1 = above[i - 1], above[i]          # m0 >= target >= m1
    f = 0.0 if m0 == m1 else (m0 - float(target_mass)) / (m0 - m1)
    return float(edges[i - 1] + f * (edges[i] - edges[i - 1]))


def _downsample(vol: np.ndarray, size: int) -> tuple[np.ndarray, int]:
    """Stride-downsample so the largest edge is <= `size` (cheap, dependency-free).

    Returns the strided grid and the stride, so callers can place it in the very same
    normalized cube `mesh_at` uses (see `grid_box`)."""
    step = max(1, int(np.ceil(max(vol.shape) / float(size))))
    return np.ascontiguousarray(vol[::step, ::step, ::step], dtype=np.float32), step


def grid_box(full_shape, ds_shape, step: int, full_size: int | None = None, offset=None):
    """Model-space corners of a strided grid inside the `[-1,1]` cube `mesh_at`
    normalizes into, so the volume ray-marcher and the marching-cubes surface frame
    the map identically.

    `mesh_at` maps voxel `i` (any axis) to `(i + off - S/2)/(S/2)`, `S = max(full_shape)`.
    The strided grid samples voxels `{0, step, 2*step, ...}`, so its sample `j` sits at
    `2*(j*step + off)/S - 1` — exactly where the mesh puts that voxel. Corners are
    edge-extended by half a cell so a ray-marcher recovers the texture coord with a plain
    `(p - lo)/(hi - lo)`. Returned per **model axis** = numpy axis (X=axis 0, Y=axis 1,
    Z=axis 2), matching the marching-cubes vertex order — the ray-marcher swaps to its
    own texture-axis order itself.

    `full_size` and `offset` mirror `mesh_at`'s: pass the same values there and here, or
    the surface and the ray-march will disagree about where the map is."""
    S = float(full_size if full_size else max(full_shape))
    off = [0.0, 0.0, 0.0] if offset is None else [float(o) for o in offset]
    half = step / S                                   # half a strided cell, model units
    lo = [2.0 * o / S - 1.0 - half for o in off]
    hi = [(2 * (n * step + off[a]) - step) / S - 1.0 for a, n in enumerate(ds_shape)]
    return lo, hi


def _scatter_add(idx: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """Sum `vals` (m, c) into `n` rows selected by `idx` (m,) — i.e. `np.add.at`, but fast.

    `np.add.at` is the obvious way to write this and the wrong one: it is unbuffered,
    single-threaded, and holds the GIL for its whole run. On a 320³ mesh (millions of
    faces) that alone was tens of seconds per decode, during which the viewer's own HTTP
    server could not answer anything. `np.bincount` computes the same sums in a tight C
    loop, per column.
    """
    out = np.empty((n, vals.shape[1]), dtype=np.float64)
    for c in range(vals.shape[1]):
        out[:, c] = np.bincount(idx, weights=vals[:, c], minlength=n)
    return out


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Smooth (area-weighted) per-vertex normals from face geometry."""
    nv = verts.shape[0]
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n = np.zeros((nv, 3), np.float64)
    for k in range(3):
        n += _scatter_add(faces[:, k], fn, nv)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return (n / ln).astype(np.float32)


def _taubin_smooth(verts: np.ndarray, faces: np.ndarray, iterations: int,
                   lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin λ|μ low-pass smoothing of the surface. Alternating positive/negative
    Laplacian steps relax the mesh (removing marching-cubes faceting) without the
    volume shrinkage that plain Laplacian smoothing causes.

    The neighbour-averaging is a sparse matrix product, built once and reused across the
    iterations: with `smooth=6` the Laplacian is evaluated twelve times, and doing that
    with `np.add.at` (see `_scatter_add`) was the single most GIL-toxic thing in the
    backend. `A` sums duplicate (src, dst) pairs exactly as the scatter-add did, so the
    result is unchanged.
    """
    if iterations <= 0 or len(verts) == 0 or len(faces) == 0:
        return verts
    v = verts.astype(np.float64)
    nv = v.shape[0]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)   # undirected
    src, dst = edges[:, 0], edges[:, 1]
    deg = np.bincount(src, minlength=nv).astype(np.float64)
    deg[deg == 0] = 1.0

    try:
        from scipy.sparse import csr_matrix
        # row-normalised adjacency: (A @ p)[i] = mean of p over i's neighbours
        A = csr_matrix((1.0 / deg[src], (src, dst)), shape=(nv, nv))
        laplacian = lambda p: A @ p - p                                    # noqa: E731
    except ImportError:   # scipy ships with scikit-image, but don't hard-fail on it
        laplacian = lambda p: _scatter_add(src, p[dst], nv) / deg[:, None] - p   # noqa: E731

    for _ in range(int(iterations)):
        v += lam * laplacian(v)
        v += mu * laplacian(v)
    return v.astype(np.float32)


def mesh_at(vol: np.ndarray, level: float, smooth: int = 0,
            step: int = 1, full_size: int | None = None, offset=None) -> bytes:
    """Marching-cubes mesh of `vol` at an absolute `level`, packed as binary:

    int32 nverts, int32 nfaces, float32 level, float32 mass, float32 vmax,
    then float32[nverts*3] positions, float32[nverts*3] normals,
    then uint32[nfaces*3] indices. Positions are normalized to a [-1,1] cube.
    `smooth` runs that many Taubin smoothing iterations on the surface.

    `step`/`full_size` describe a *draft* mesh taken from a strided view of a bigger
    volume (see `/api/decode?size=`): sample `j` of the strided grid is voxel `j*step` of
    the full one, so scaling the vertices by `step` and normalizing by the full size puts
    the draft in exactly the same [-1,1] cube as the full-resolution surface — it refines
    in place rather than jumping when the final mesh replaces it. Enclosed mass is scaled
    back up by `step**3` for the same reason: so a draft's reported mass is comparable to
    a full one (the client keys constant-mass morphs off it).

    `offset` shifts the vertices by that many voxels of the *full* grid before
    normalizing. One map alone never needs it (it is its own frame of reference), but
    several maps shown together do: pass a common `full_size` and a per-map `offset` and
    boxes of different sizes come out concentric and to relative scale, instead of each
    filling the cube on its own and pretending to be the same size. Default None keeps
    the single-map normalization byte-for-byte.
    """
    from skimage import measure

    vmin, vmax = float(vol.min()), float(vol.max())
    level = min(max(float(level), vmin + 1e-6), vmax - 1e-6)
    try:
        verts, faces, normals, _ = measure.marching_cubes(vol, level=level)
    except (ValueError, RuntimeError):
        verts = np.zeros((0, 3), np.float32)
        faces = np.zeros((0, 3), np.int64)
        normals = np.zeros((0, 3), np.float32)

    if smooth and faces.shape[0] > 0:
        verts = _taubin_smooth(verts, faces, smooth)
        normals = _vertex_normals(verts, faces)   # recompute from the relaxed surface

    size = float(full_size if full_size else max(vol.shape))
    off = np.zeros(3, np.float32) if offset is None else np.asarray(offset, np.float32)
    verts = (verts.astype(np.float32) * float(step) + off - size / 2.0) / (size / 2.0)

    header = (
        np.array([verts.shape[0], faces.shape[0]], dtype="<i4").tobytes()
        + np.array([level, enclosed_mass(vol, level) * float(step) ** 3, vmax], dtype="<f4").tobytes()
    )
    body = (
        np.ascontiguousarray(verts, dtype="<f4").tobytes()
        + np.ascontiguousarray(normals, dtype="<f4").tobytes()
        + np.ascontiguousarray(faces, dtype="<u4").tobytes()
    )
    return header + body


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
class Exporter:
    """Writes per-layer outputs (subset metadata + average-latent volume) into the
    owning project. Only available when the viewer is launched with a project dir
    (and, for particle metadata, a row-aligned `--metadata`)."""

    def __init__(self, decoder: Decoder, metadata: str | None, project_dir: str | None) -> None:
        self.decoder = decoder
        self.metadata_path = metadata
        self.project_dir = Path(project_dir).expanduser() if project_dir else None
        self._md = None  # lazy XmippMetaData

    @property
    def available(self) -> bool:
        return self.project_dir is not None

    def _md_table(self):
        if self._md is None and self.metadata_path:
            from xmipp_metadata.metadata import XmippMetaData
            self._md = XmippMetaData(self.metadata_path)
        return self._md

    def save_layer(self, name: str, indices: list[int], fmt: str) -> dict:
        import re
        import numpy as np
        from xmipp_metadata.image_handler import ImageHandler

        if self.project_dir is None:
            raise RuntimeError("export needs a project directory (open the viewer from a job)")
        idx = [int(i) for i in indices if 0 <= int(i) < self.decoder.total]
        if not idx:
            raise RuntimeError("layer has no valid states to save")
        safe = re.sub(r"[^\w.-]+", "_", str(name)).strip("_") or "layer"
        folder = self.project_dir / safe
        folder.mkdir(parents=True, exist_ok=True)
        files: list[str] = []

        # volume decoded from the average latent of the whole layer
        avg = np.mean([self.decoder._latent(i) for i in idx], axis=0)
        vol = self.decoder._decode_latent(avg)
        vol_name = f"{safe}_average.mrc"
        ImageHandler().write(vol, str(folder / vol_name), overwrite=True)
        files.append(vol_name)

        # subset metadata for the real particles in the layer (centroids have no row)
        real = [i for i in idx if i < self.decoder.landscape.n]
        n_particles = 0
        md = self._md_table()
        if md is not None and real:
            ext = fmt if fmt in {"xmd", "star", "cs"} else "xmd"
            from xmipp_metadata.metadata import XmippMetaData
            sub = XmippMetaData()
            sub.table = md.table.iloc[real].reset_index(drop=True)
            md_name = f"{safe}.{ext}"
            sub.write(str(folder / md_name))
            files.append(md_name)
            n_particles = len(real)
        return {"folder": str(folder), "files": files, "n": len(idx), "n_particles": n_particles}


def _parse_seeds(seeds: str) -> list[int] | None:
    """"3,17,42" -> [3, 17, 42]; empty or malformed -> None (i.e. unseeded KMeans)."""
    if not seeds.strip():
        return None
    try:
        return [int(x) for x in seeds.split(",") if x.strip() != ""]
    except ValueError:
        return None


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

    @app.get("/api/landscape")
    def landscape() -> dict:
        """Just the metadata. The arrays are fetched separately, as binary."""
        return {
            "title": title,
            "n": int(ls.n),
            "dims": int(ls.dim),
            "pca_var": ls.pca_var,
            "pca_k": int(ls.pca.shape[1]),
        }

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
            raise HTTPException(status_code=404, detail="latent index out of range")
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
            raise HTTPException(status_code=404, detail="latent index out of range")
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
            raise HTTPException(status_code=404, detail="latent index out of range")
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
            raise HTTPException(status_code=404, detail="latent index out of range")
        with _DECODE_SLOTS:
            try:
                grid, step = (_draft_grid(index, None, 0.0, int(size)) if size > 0
                              else (_resolve_volume(index, None, 0.0), 1))
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"decode failed: {exc}")
            level = level_for_sigma(grid, sigma)
            # Scale back up by the stride so a draft's mass is comparable to a full one —
            # the same convention `mesh_at` uses for the mass it reports in its header.
            return {"level": level, "mass": enclosed_mass(grid, level) * float(step) ** 3}

    @app.get("/api/histogram")
    def histogram(index: int = Query(...), bins: int = Query(96)) -> dict:
        if index < 0 or index >= decoder.total:
            raise HTTPException(status_code=404, detail="latent index out of range")
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
            raise HTTPException(status_code=404, detail="latent index out of range")
        import tempfile
        from xmipp_metadata.image_handler import ImageHandler

        vol = decoder.volume(index)
        with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as tf:
            path = tf.name
        try:
            ImageHandler().write(vol, path, overwrite=True)
            data = Path(path).read_bytes()
        finally:
            Path(path).unlink(missing_ok=True)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="state_{index:04d}.mrc"'},
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
