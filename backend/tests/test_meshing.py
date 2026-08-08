"""Regression tests for the viewer's mesh/grid axis conventions.

The one that matters most is chirality: marching_cubes returns vertices in
array-index order (z, y, x for an MRC-convention volume), and rendering those
directly as model (x, y, z) mirror-images every map (left-handed alpha-helices).
`mesh_at` now flips to spatial order; these tests pin that down so it can never
silently regress.

Run with pytest, or directly:  python backend/tests/test_meshing.py
(needs the hax env: numpy + scikit-image, no JAX required for these paths)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hax_gui" / "viewer_app"))

from server import _downsample, grid_box, mesh_at  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def parse_mesh(buf: bytes):
    """Unpack mesh_at's wire format (see its docstring)."""
    nverts, nfaces = (int(v) for v in np.frombuffer(buf[:8], "<i4"))
    off = 8 + 12  # header: 2*i4 + 3*f4 (level, mass, vmax)
    verts = np.frombuffer(buf, "<f4", nverts * 3, off).reshape(-1, 3)
    off += nverts * 12
    normals = np.frombuffer(buf, "<f4", nverts * 3, off).reshape(-1, 3)
    off += nverts * 12
    faces = np.frombuffer(buf, "<u4", nfaces * 3, off).reshape(-1, 3)
    return verts, normals, faces


def blob_volume(shape_zyx, center_xyz, sigma=2.0):
    """A gaussian blob in a numpy (z, y, x)-ordered volume at MRC (x, y, z)."""
    nz, ny, nx = shape_zyx
    cx, cy, cz = center_xyz
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return np.exp(-r2 / (2.0 * sigma**2)).astype(np.float32)


def to_voxels(verts, full_size):
    """Model [-1,1] coords back to voxel units of the full grid (x, y, z)."""
    h = full_size / 2.0
    return verts * h + h


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_axis_identity():
    """A blob at MRC (x=24, y=16, z=8) must mesh at model (24, 16, 8).

    Under the pre-fix transposition the centroid came out at (8, 16, 24):
    the x and z axes swapped, i.e. the mirror bug."""
    vol = blob_volume((32, 32, 32), center_xyz=(24, 16, 8))
    verts, _, _ = parse_mesh(mesh_at(vol, level=0.5))
    assert len(verts) > 0
    c = to_voxels(verts, 32).mean(axis=0)
    assert abs(c[0] - 24) < 1.5, f"model x should be MRC x=24, got {c}"
    assert abs(c[1] - 16) < 1.5, f"model y should be MRC y=16, got {c}"
    assert abs(c[2] - 8) < 1.5, f"model z should be MRC z=8, got {c}"


def _twist_slope(verts):
    """Signed twist of a tubular vertex cloud about its own principal axis.

    Chirality invariant: unchanged by any rotation or by flipping the axis
    direction, but its SIGN flips under any mirror/transposition of the
    geometry. So it detects the old (z,y,x)->(x,y,z) transposition no matter
    which model axis the helix ends up lying along."""
    c = verts.mean(axis=0)
    p = verts - c
    # principal axis of the cloud = helix axis (the helix is built long & thin)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    d = vt[0]
    # right-handed frame (e1, e2, d)
    seed = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    t = p @ d
    phi = np.arctan2(p @ e2, p @ e1)
    lo, hi = np.quantile(t, 0.08), np.quantile(t, 0.92)  # drop the capped ends
    bins = np.linspace(lo, hi, 20)
    idx = np.digitize(t, bins)
    tc, ang = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() < 5:
            continue
        tc.append(0.5 * (bins[b - 1] + bins[b]))
        ang.append(np.arctan2(np.sin(phi[m]).mean(), np.cos(phi[m]).mean()))
    tc, ang = np.asarray(tc), np.unwrap(np.asarray(ang))
    return float(np.polyfit(tc, ang, 1)[0])


def test_chirality_right_handed_helix():
    """A right-handed helix must stay right-handed through meshing.

    Build a tube whose angle about +z increases with z (counterclockwise seen
    from +z: right-handed, like an alpha-helix) and check the mesh's intrinsic
    twist sign. Any det=-1 transform of the vertices (the old transposition, or
    a single-axis flip introduced later) makes it negative."""
    n, cx, cy, R, rate = 64, 32.0, 32.0, 12.0, 0.12
    vol = np.zeros((n, n, n), np.float32)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for z in range(6, 58):
        th = rate * (z - 6)
        px, py = cx + R * np.cos(th), cy + R * np.sin(th)
        vol[z] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * 2.5**2))
    verts, _, _ = parse_mesh(mesh_at(vol, level=0.5))
    assert len(verts) > 100
    slope = _twist_slope(to_voxels(verts, n))
    assert slope > 0.05, (
        f"helix handedness flipped: twist = {slope:.3f} rad/voxel (built ~+{rate})"
    )
    # the invariant itself must flip on mirrored geometry, or this test is blind
    assert _twist_slope(to_voxels(verts, n)[:, ::-1]) < -0.05


def test_offset_in_numpy_order():
    """Multi-map framing: `offset` is given in numpy (z, y, x) axis order and
    must land the map at the right place in the model (x, y, z) cube."""
    shape = (16, 32, 48)  # nz, ny, nx
    full = 48
    off_numpy = [(full - s) / 2.0 for s in shape]  # centred, as _Scene.norm does
    vol = blob_volume(shape, center_xyz=(40, 16, 8))
    verts, _, _ = parse_mesh(mesh_at(vol, level=0.5, full_size=full, offset=off_numpy))
    c = to_voxels(verts.mean(axis=0), full)
    # x: 40 + 0 = 40 ; y: 16 + 8 = 24 (centre) ; z: 8 + 16 = 24 (centre)
    assert abs(c[0] - 40) < 1.5, f"got {c}"
    assert abs(c[1] - 24) < 1.5, f"got {c}"
    assert abs(c[2] - 24) < 1.5, f"got {c}"


def test_grid_box_model_order():
    """grid_box takes numpy-order shapes/offsets and returns model (x, y, z)
    corners: spans must scale with the (nx, ny, nz) axis lengths and a centred
    map's box must be centred."""
    shape = (16, 32, 48)
    full = 48
    off_numpy = [(full - s) / 2.0 for s in shape]
    vol = np.zeros(shape, np.float32)
    ds, step = _downsample(vol, 24)
    lo, hi = grid_box(shape, ds.shape, step, full_size=full, offset=off_numpy)
    spans = [h - l for l, h in zip(lo, hi)]
    mids = [(h + l) / 2.0 for l, h in zip(lo, hi)]
    assert spans[0] > spans[1] > spans[2], f"model x>y>z spans expected, got {spans}"
    assert all(abs(m) < 0.1 for m in mids), f"centred map, got mids {mids}"


def test_normal_orientation_consistent():
    """Attribute normals (smooth=0) point outward; winding-derived normals
    (smooth>0, `_vertex_normals`) have the sense the renderer has always been
    shown (inward — gl.js lights two-sided via gl_FrontFacing, so only
    *consistency* matters). If either assertion flips, the winding and the
    normal transform in `mesh_at` have gone out of sync — check both together
    with gl.js before touching this."""
    vol = blob_volume((32, 32, 32), center_xyz=(16, 16, 16), sigma=5.0)
    for smooth, expect_outward in ((0, True), (3, False)):
        verts, normals, _ = parse_mesh(mesh_at(vol, level=0.5, smooth=smooth))
        c = verts.mean(axis=0)
        d = ((verts - c) * normals).sum(axis=1)
        frac_out = float((d > 0).mean())
        if expect_outward:
            assert frac_out > 0.95, f"smooth={smooth}: expected outward, frac={frac_out}"
        else:
            assert frac_out < 0.05, f"smooth={smooth}: expected inward, frac={frac_out}"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if fails else 0)
