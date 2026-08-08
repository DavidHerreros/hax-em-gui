"""Latent-space landscape: embeddings (PCA + UMAP), clustering, filtering.

Extracted from server.py; everything latents-side and JAX-free lives here.
"""

from __future__ import annotations

import os
import time

import numpy as np

from vlog import log_exc as _log_exc

def _knn_distances(x: np.ndarray, k: int) -> tuple[np.ndarray, str]:
    """(n, k) distances from each row of `x` to its k nearest rows, itself included.

    Tries the GPU first — cuML is what ``hax.utils.filter_latent_space`` uses when JAX
    has a device — then falls back to exact CPU searches. The device is whatever this
    process was pinned to at spawn (see ``select_gpu``); an empty CUDA_VISIBLE_DEVICES
    means the viewer was deliberately started on the CPU, so don't build a CUDA context
    behind its back. Nothing here imports JAX: asking ``jax.default_backend()`` which
    path to take, as the batch program does, would initialise a backend this process
    defers until the first decode.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") != "":
        try:
            from cuml.neighbors.nearest_neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(x)
            dist, _idx = nn.kneighbors(x)
            return np.asarray(dist, dtype=np.float32), "cuML (GPU)"
        except ImportError:
            print("[viewer] cuML not available; running the k-NN on the CPU", flush=True)
        except Exception:  # noqa: BLE001 - a GPU that refuses is not a reason to fail
            _log_exc("cuML k-NN failed; falling back to the CPU")
    try:
        from scipy.spatial import cKDTree
        dist, _idx = cKDTree(x).query(x, k=k, workers=-1)
        return np.asarray(dist, dtype=np.float32), "scipy cKDTree"
    except ImportError:
        from sklearn.neighbors import NearestNeighbors as SkNN
        dist, _idx = SkNN(n_neighbors=k, n_jobs=-1).fit(x).kneighbors(x)
        return np.asarray(dist, dtype=np.float32), "sklearn"

class Landscape:
    def __init__(self, latents: np.ndarray, decode: np.ndarray | None = None) -> None:
        # `latents` drives the display (embedding/clustering); `decode` is the
        # full latent space whose rows decode to volumes. They differ only when a
        # reduced display space (z_space_reduced) is supplied — then centroids are
        # returned in the decodable full space so generated states still work.
        self._all = latents
        self._all_decode = decode if decode is not None else latents
        self.set_filter(None)

    def set_filter(self, rows: np.ndarray | None) -> None:
        """Restrict the landscape to `rows` of the original set (None = all of it).

        Everything downstream of here — the embeddings, the clustering, the snap trees,
        the z-scores — is *fitted*, so a filtered landscape is not the old one with points
        hidden: it is a new PCA basis and a new UMAP graph computed without the states that
        were dropped. That refit is the entire reason to filter at all. Outliers stretch
        the principal axes toward themselves and join the k-NN graph UMAP is built from, so
        they distort the picture for every state that stays.

        The index space the API speaks in is always this filtered one: `rows` maps it back
        to the original particles, and only the two places that leave the viewer — decoding
        a volume and exporting metadata — ever need that map. Nothing in the browser knows
        a filter happened beyond the state count going down.
        """
        n_all = self._all.shape[0]
        self.rows = np.arange(n_all) if rows is None else np.asarray(rows, dtype=np.int64)
        # Identity keeps the originals rather than copying them: the unfiltered case is
        # the common one and these are the two biggest arrays in the process.
        full = rows is None
        self.latents = self._all if full else self._all[self.rows]
        self.decode = self._all_decode if full else self._all_decode[self.rows]
        self.n, self.dim = self.latents.shape
        self.n_all = n_all
        self._pca = None
        self._pca_var = None
        self._pca_comp = None     # (k, dim) basis, for projecting new points (centroids)
        self._pca_mean = None
        self._umap = None
        self._clusters: dict[int, list[int]] = {}
        self._kmeans: dict[tuple, tuple] = {}   # (k, seeds, space) -> (labels, centroids, cpca)
        self._trees: dict[str, object] = {}     # embed -> cKDTree, for snapping drawn curves
        self._zscores: dict[int, np.ndarray] = {}   # k -> per-state outlier z-score

    @property
    def filtered(self) -> bool:
        return self.n != self.n_all

    def filter_by_zscore(self, thr: float, k: int) -> int:
        """Keep the states whose neighbourhood z-score is below `thr`; returns how many.

        Measured on the *current* set, so applying a filter twice narrows what is already
        there rather than starting over — the z-scores are re-derived from the states that
        survived, which is the same thing running `filter_latents` on its own output would
        give you.
        """
        keep = np.flatnonzero(self.zscores(k) < float(thr))
        if keep.size == 0:
            raise ValueError("that threshold keeps no states")
        self.set_filter(self.rows[keep])
        return self.n

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

    def zscores(self, k: int) -> np.ndarray:
        """|z| of each state's mean distance to its k nearest neighbours.

        The statistic ``filter_latents`` thresholds (``hax.utils.filter_latent_space``): a
        state sitting in a sparse part of the space has an unusually large mean neighbour
        distance, so a large |z| marks it as an outlier. Measured in the display latents —
        the coordinates the clustering also uses, never the PCA/UMAP projection, which is
        display-only.

        Memoised per k, and that memo is the point of doing this here at all: the
        neighbour search is the entire cost, while the threshold is a comparison against
        this vector that the browser does for itself. Moving ``k`` refits; moving the
        threshold does not, so it stays interactive at any N.

        The mean is over the k-1 rows ``kneighbors(k)`` returns besides the point itself —
        deliberately the same off-by-one as ``filter_latent_space``, so a threshold tuned
        here means the same thing when it is handed to ``filter_latents --thr``.
        """
        k = int(max(2, min(200, k, self.n - 1)))
        hit = self._zscores.get(k)
        if hit is not None:
            return hit
        t0 = time.time()
        x = np.ascontiguousarray(self.latents, dtype=np.float32)
        dist, backend = _knn_distances(x, k)
        mean_d = dist[:, 1:].mean(axis=1)          # column 0 is the point itself, at distance 0
        std = float(mean_d.std())
        z = np.abs((mean_d - float(mean_d.mean())) / (std if std > 0 else 1.0)).astype(np.float32)
        print(f"[viewer] outlier z-scores: k={k} n={self.n} dim={self.dim} "
              f"via {backend} in {time.time() - t0:.2f}s", flush=True)
        self._zscores[k] = z
        return z

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
