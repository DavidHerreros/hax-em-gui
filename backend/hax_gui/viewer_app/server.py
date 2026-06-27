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
from pathlib import Path

# Be a good GPU citizen: don't preallocate the whole device.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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


# --------------------------------------------------------------------------
# Landscape: embeddings (PCA/UMAP) + clustering of the latent space.
# --------------------------------------------------------------------------
class Landscape:
    def __init__(self, latents: np.ndarray) -> None:
        self.latents = latents
        self.n, self.dim = latents.shape
        self._pca = None
        self._pca_var = None
        self._pca_comp = None     # (k, dim) basis, for projecting new points (centroids)
        self._pca_mean = None
        self._umap = None
        self._clusters: dict[int, list[int]] = {}

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

    def kmeans_full(self, k: int):
        """KMeans on the latents -> (labels, centroid latents, centroid PCA coords)."""
        k = int(max(2, min(20, k)))
        _ = self.pca  # ensure the PCA basis exists
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(self.latents)
            labels = km.labels_.astype(int)
            centroids = km.cluster_centers_.astype(np.float32)
        except Exception:  # noqa: BLE001 - degenerate fallback: one cluster
            labels = np.zeros(self.n, dtype=int)
            centroids = self.latents.mean(0, keepdims=True).astype(np.float32)
        cpca = ((centroids - self._pca_mean) @ self._pca_comp.T).astype(np.float32)
        return labels, centroids, cpca

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

    def clusters(self, k: int) -> list[int]:
        k = int(max(1, min(20, k)))
        if k not in self._clusters:
            try:
                from sklearn.cluster import KMeans
                labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(self.latents)
            except Exception:  # noqa: BLE001
                labels = np.zeros(self.n, dtype=int)
            self._clusters[k] = [int(v) for v in labels]
        return self._clusters[k]


# --------------------------------------------------------------------------
# Decoder: load network + latents, decode a latent -> density grid (lazy JAX).
# --------------------------------------------------------------------------
class Decoder:
    def __init__(self, latents_path: str, reload_path: str) -> None:
        self.latents = _load_latents(latents_path)
        self.landscape = Landscape(self.latents)
        self._reload = reload_path
        self._decode = None  # built lazily on first decode (defers the JAX import)
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
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
        return self.landscape.n

    def _latent(self, index: int) -> np.ndarray:
        n = self.landscape.n
        if 0 <= index < n:
            return self.latents[index]
        if n <= index < n + len(self.extra):
            return self.extra[index - n]
        raise IndexError(f"latent index {index} out of range")

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

    def _decode_latent(self, latent: np.ndarray) -> np.ndarray:
        self._ensure_model()
        x = self._jnp.asarray(latent)
        # np.array (not asarray) forces a writable copy -- JAX arrays are
        # read-only and skimage.marching_cubes needs a writable buffer.
        vol = np.array(self._decode(x), dtype=np.float32)
        vol = np.squeeze(vol)
        if vol.ndim == 4:
            vol = vol[0]
        if vol.ndim != 3:
            raise RuntimeError(f"decode_volume returned shape {vol.shape}, expected a 3D grid")
        return vol

    def volume(self, index: int) -> np.ndarray:
        if index in self._cache:
            return self._cache[index]
        vol = self._decode_latent(self._latent(index))
        self._cache[index] = vol
        self._cache_order.append(index)
        while len(self._cache_order) > 8:  # cap memory: keep the 8 most recent volumes
            self._cache.pop(self._cache_order.pop(0), None)
        return vol

    def volume_interp(self, a: int, b: int, t: float) -> np.ndarray:
        """Decode the latent interpolated between states a and b (for morphs)."""
        latent = (1.0 - t) * self._latent(a) + t * self._latent(b)
        return self._decode_latent(latent)


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


def level_for_mass(vol: np.ndarray, target_mass: float) -> float:
    """Bisection for the level whose enclosed mass equals `target_mass`.

    Mass is monotonically non-increasing in level, so a simple binary search
    converges. Used to morph at constant mass (a fair shape comparison)."""
    lo, hi = float(vol.min()), float(vol.max())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if enclosed_mass(vol, mid) > target_mass:
            lo = mid  # too much mass -> raise the level
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _downsample(vol: np.ndarray, size: int) -> np.ndarray:
    """Stride-downsample so the largest edge is <= `size` (cheap, dependency-free)."""
    step = max(1, int(np.ceil(max(vol.shape) / float(size))))
    return np.ascontiguousarray(vol[::step, ::step, ::step], dtype=np.float32)


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Smooth (area-weighted) per-vertex normals from face geometry."""
    n = np.zeros(verts.shape, np.float64)
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return (n / ln).astype(np.float32)


def _taubin_smooth(verts: np.ndarray, faces: np.ndarray, iterations: int,
                   lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin λ|μ low-pass smoothing of the surface. Alternating positive/negative
    Laplacian steps relax the mesh (removing marching-cubes faceting) without the
    volume shrinkage that plain Laplacian smoothing causes."""
    if iterations <= 0 or len(verts) == 0 or len(faces) == 0:
        return verts
    v = verts.astype(np.float64)
    nv = v.shape[0]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)   # undirected
    src, dst = edges[:, 0], edges[:, 1]
    deg = np.bincount(src, minlength=nv).astype(np.float64)
    deg[deg == 0] = 1.0

    def laplacian(p):
        acc = np.zeros_like(p)
        np.add.at(acc, src, p[dst])
        return acc / deg[:, None] - p

    for _ in range(int(iterations)):
        v += lam * laplacian(v)
        v += mu * laplacian(v)
    return v.astype(np.float32)


def mesh_at(vol: np.ndarray, level: float, smooth: int = 0) -> bytes:
    """Marching-cubes mesh of `vol` at an absolute `level`, packed as binary:

    int32 nverts, int32 nfaces, float32 level, float32 mass, float32 vmax,
    then float32[nverts*3] positions, float32[nverts*3] normals,
    then uint32[nfaces*3] indices. Positions are normalized to a [-1,1] cube.
    `smooth` runs that many Taubin smoothing iterations on the surface.
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

    size = float(max(vol.shape))
    verts = (verts.astype(np.float32) - size / 2.0) / (size / 2.0)

    header = (
        np.array([verts.shape[0], faces.shape[0]], dtype="<i4").tobytes()
        + np.array([level, enclosed_mass(vol, level), vmax], dtype="<f4").tobytes()
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


def build_app(decoder: Decoder, title: str = "Conformational landscape",
              metadata: str | None = None, project_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Hax WebGL viewer")
    ls = decoder.landscape
    exporter = Exporter(decoder, metadata, project_dir)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "n": int(ls.n), "dims": int(ls.dim)}

    @app.get("/api/landscape")
    def landscape(k: int = Query(5)) -> dict:
        return {
            "title": title,
            "n": int(ls.n),
            "dims": int(ls.dim),
            "pca": ls.pca.tolist(),
            "pca_var": ls.pca_var,
            "pca_k": int(ls.pca.shape[1]),
            "latents": ls.latents.tolist(),   # raw latent coords, for the "latent" view
            "cluster": {"k": int(k), "labels": ls.clusters(k)},
        }

    @app.get("/api/embedding")
    def embedding(kind: str = Query("umap")) -> dict:
        if kind == "umap":
            return {"kind": "umap", "points": ls.umap().tolist()}
        return {"kind": "pca", "points": ls.pca[:, :2].tolist()}

    @app.get("/api/cluster")
    def cluster(k: int = Query(5)) -> dict:
        return {"k": int(k), "labels": ls.clusters(k)}

    @app.get("/api/kmeans")
    def kmeans(k: int = Query(5)) -> dict:
        """KMeans -> per-state labels + decodable centroids (virtual indices base..base+k-1)."""
        labels, centroids, cpca = ls.kmeans_full(k)
        base = decoder.set_centroids(centroids)
        return {
            "k": int(centroids.shape[0]),
            "labels": [int(v) for v in labels],
            "base": int(base),
            "centroid_pca": cpca.tolist(),
        }

    def _resolve_volume(index: int, index2: int | None, t: float) -> np.ndarray:
        if index < 0 or index >= decoder.total:
            raise HTTPException(status_code=404, detail="latent index out of range")
        if index2 is not None and 0 <= index2 < decoder.total and index2 != index:
            return decoder.volume_interp(index, index2, max(0.0, min(1.0, t)))
        return decoder.volume(index)

    @app.get("/api/decode")
    def decode(
        index: int = Query(...),
        sigma: float = Query(2.5),
        mass: float = Query(-1.0),       # >=0 -> contour at constant enclosed mass
        index2: int = Query(-1),         # >=0 -> interpolate index..index2 at t
        t: float = Query(0.0),
        smooth: int = Query(0),          # Taubin smoothing iterations on the surface
    ) -> Response:
        try:
            vol = _resolve_volume(index, index2 if index2 >= 0 else None, t)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"decode failed: {exc}")
        level = level_for_mass(vol, mass) if mass >= 0 else level_for_sigma(vol, sigma)
        return Response(content=mesh_at(vol, level, smooth=max(0, min(50, smooth))), media_type="application/octet-stream")

    @app.get("/api/grid")
    def grid(
        index: int = Query(...),
        index2: int = Query(-1),
        t: float = Query(0.0),
        size: int = Query(96),       # max grid edge of the streamed texture
    ) -> Response:
        """Decoded density as a compact uint8 3D-texture payload for ray-marching.

        Layout: int32 nx, ny, nz (texture width/height/depth, x fastest), then
        float32 vmin, vmax, mean, std, then nx*ny*nz uint8 voxels (C order)."""
        try:
            vol = _resolve_volume(index, index2 if index2 >= 0 else None, t)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"decode failed: {exc}")
        ds = _downsample(vol, int(max(16, min(160, size))))
        vmin, vmax = float(ds.min()), float(ds.max())
        u8 = np.clip((ds - vmin) / (vmax - vmin + 1e-9) * 255.0, 0, 255).astype(np.uint8)
        # numpy C order has the last axis varying fastest -> that is the texture x.
        nx, ny, nz = ds.shape[2], ds.shape[1], ds.shape[0]
        header = (
            np.array([nx, ny, nz], dtype="<i4").tobytes()
            + np.array([vmin, vmax, float(vol.mean()), float(vol.std())], dtype="<f4").tobytes()
        )
        return Response(content=header + np.ascontiguousarray(u8).tobytes(), media_type="application/octet-stream")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="WebGL conformational-landscape viewer backend")
    parser.add_argument("--latents", required=True, help="latents .npy/.txt of shape (N, latent_dim)")
    parser.add_argument("--reload", required=True, help="trained network checkpoint dir (e.g. <run>/HetSIREN)")
    parser.add_argument("--title", default="Conformational landscape")
    parser.add_argument("--metadata", default=None, help="row-aligned particle metadata (.xmd/.star) for layer export")
    parser.add_argument("--project-dir", dest="project_dir", default=None, help="project folder where exported layers are written")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    args = parser.parse_args()

    decoder = Decoder(args.latents, args.reload)
    app = build_app(decoder, title=args.title, metadata=args.metadata, project_dir=args.project_dir)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
