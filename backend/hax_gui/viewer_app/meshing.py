"""Server-side mesh extraction: contour levels, downsampling, marching cubes.

Pure numpy + scikit-image — no JAX, no hax. Shared by the landscape viewer
(server.py) and the standalone map viewer (mrc_server.py), which used to reach
into the JAX-adjacent server module just for these functions.
"""

from __future__ import annotations

import numpy as np

import wirefmt

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
    `(p - lo)/(hi - lo)`. Inputs (`full_shape`, `ds_shape`, `offset`) are in **numpy axis
    order**; the returned corners are in **model (x, y, z) order** — numpy axes reversed —
    matching the flipped marching-cubes vertices (see `mesh_at`), so the ray-marcher
    samples the texture with no swizzle: normalized model coords are texture (s, t, r).

    `full_size` and `offset` mirror `mesh_at`'s: pass the same values there and here, or
    the surface and the ray-march will disagree about where the map is."""
    S = float(full_size if full_size else max(full_shape))
    off = [0.0, 0.0, 0.0] if offset is None else [float(o) for o in offset]
    half = step / S                                   # half a strided cell, model units
    lo = [2.0 * o / S - 1.0 - half for o in off]
    hi = [(2 * (n * step + off[a]) - step) / S - 1.0 for a, n in enumerate(ds_shape)]
    return lo[::-1], hi[::-1]

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
    then uint32[nfaces*3] indices. Positions are normalized to a [-1,1] cube, in
    spatial (x, y, z) order — numpy axes reversed, see the flip below.
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

    # marching_cubes returns coordinates in array-index order (axis0, axis1, axis2),
    # which for an MRC-convention volume is (z, y, x). Rendering that directly as
    # model (x, y, z) is a transposition (det = -1): every map came out mirror-imaged
    # (left-handed alpha-helices). Flip to spatial (x, y, z) order, and reverse the
    # triangle winding so the faces' geometric orientation stays consistent with the
    # flipped normals (`_vertex_normals` re-derives normals from winding after
    # smoothing, and a mirror alone would flip their sense).
    verts = verts[:, ::-1]
    normals = normals[:, ::-1]
    faces = faces[:, ::-1]

    if smooth and faces.shape[0] > 0:
        verts = _taubin_smooth(verts, faces, smooth)
        normals = _vertex_normals(verts, faces)   # recompute from the relaxed surface

    size = float(full_size if full_size else max(vol.shape))
    # `offset` arrives in numpy axis order (callers derive it from vol.shape);
    # the vertices were just flipped to spatial order, so flip it to match.
    off = np.zeros(3, np.float32) if offset is None else np.asarray(offset, np.float32)[::-1]
    verts = (verts.astype(np.float32) * float(step) + off - size / 2.0) / (size / 2.0)

    header = (
        wirefmt.header(wirefmt.MESH)
        + np.array([verts.shape[0], faces.shape[0]], dtype="<i4").tobytes()
        + np.array([level, enclosed_mass(vol, level) * float(step) ** 3, vmax], dtype="<f4").tobytes()
    )
    body = (
        np.ascontiguousarray(verts, dtype="<f4").tobytes()
        + np.ascontiguousarray(normals, dtype="<f4").tobytes()
        + np.ascontiguousarray(faces, dtype="<u4").tobytes()
    )
    return header + body
