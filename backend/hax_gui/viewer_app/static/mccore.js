// Contouring, in the browser. Pure functions over typed arrays — no DOM, no WebGL, no
// dependencies — so this module runs unchanged in the page, in a Web Worker, and under
// plain `node mccore.test.mjs`, which is what makes it testable at all.
//
// Everything here mirrors a piece of the backend's meshing.py, deliberately and to the
// letter, because a locally contoured surface has to land in exactly the same place as a
// server one: a draft fetched from the server must refine into a local full-resolution mesh
// without the map appearing to shift, and the surface has to register with the ray-marched
// iso view of the same grid. The two rules that guarantee that are:
//
//   * sample j of a (possibly strided) grid sits at  lo + (j + 0.5) * (hi - lo) / n,
//     which is meshing.grid_box's mapping solved for the sample centre — the same
//     mapping the ray-marcher inverts with (p - lo) / (hi - lo);
//   * normals point along MINUS the density gradient (outward from a high-density blob),
//     which is what skimage's 'descent' orientation gives the server after its axis flip.
//
// Marching TETRAHEDRA, not marching cubes. A deliberate deviation, for two reasons. The
// classic Bourke cube tables are 4096 hand-transcribed integers with no reference to diff
// against here, and a single wrong entry shows up as a hole in a mesh nobody would notice
// until a figure was already made. And plain cube tables have genuinely ambiguous face
// configurations that leave holes; the server does not suffer from them because scikit-image
// defaults to the topologically-correct Lewiner variant. A tetrahedral decomposition has no
// ambiguous cases at all: every cube splits into six tetrahedra around one body diagonal,
// neighbouring cubes agree on the shared face diagonal by construction, and the result is
// watertight by construction rather than by transcription. The cost is roughly 1.7x the
// triangles for the same vertices, which Taubin smoothing and the GPU both shrug at.

// ---------------------------------------------------------------------------
// half-precision decode
// ---------------------------------------------------------------------------

// IEEE half -> float, as a 65536-entry lookup. Building it costs ~1 ms and turns every
// later decode into an array index; grids arrive as raw half bits (see wire.js), because
// that is what WebGL wants and JS has no half-precision typed array to widen them into.
let HALF_LUT = null;
export function halfToFloatTable() {
  if (HALF_LUT) return HALF_LUT;
  const t = new Float32Array(65536);
  const b = new ArrayBuffer(4), u32 = new Uint32Array(b), f32 = new Float32Array(b);
  for (let h = 0; h < 65536; h++) {
    const sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    if (exp === 0) {
      if (man === 0) { t[h] = sign ? -0 : 0; continue; }
      exp = 1;                                   // subnormal: renormalise into float32
      while (!(man & 0x400)) { man <<= 1; exp--; }
      man &= 0x3ff;
      u32[0] = sign | ((exp + 127 - 15) << 23) | (man << 13);
    } else if (exp === 31) {
      u32[0] = sign | 0x7f800000 | (man << 13);  // inf / nan
    } else {
      u32[0] = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    t[h] = f32[0];
  }
  HALF_LUT = t;
  return t;
}

// A grid payload (wire.js parseGrid) -> the normalized 0..1 field, as float32. The payload's
// vmin/vmax say what those numbers mean in the map's own units; nothing here needs them.
export function decodeField(g) {
  const n = g.nx * g.ny * g.nz, out = new Float32Array(n);
  if (g.f16) {
    const T = halfToFloatTable();
    for (let i = 0; i < n; i++) out[i] = T[g.data[i]];
  } else {
    for (let i = 0; i < n; i++) out[i] = g.data[i] / 255;
  }
  return out;
}

// The density blend a morph frame shows, in the NEAR map's normalized space: the same
// convention the ray-marcher's uBScale/uBOff implement, so the surface morph and the iso
// morph of the same pair show the same density.
export function blendFields(a, b, t, scale = 1, off = 0) {
  const n = a.length, out = new Float32Array(n), u = 1 - t;
  for (let i = 0; i < n; i++) out[i] = a[i] * u + (b[i] * scale + off) * t;
  return out;
}

// ---------------------------------------------------------------------------
// enclosed mass (the constant-mass morph's currency)
// ---------------------------------------------------------------------------

// "Mass" is meshing.enclosed_mass: the SUM OF RAW VOXEL VALUES at or above the level — not a
// count, not sum(v - level). Mirrored so a locally contoured constant-mass morph is cut at
// the same levels the server's would be.
export const MASS_BINS = 1024;

// Reverse-cumulative mass by normalized level: above[b] is the mass at or above level
// b / MASS_BINS, which is monotone non-increasing and so invertible by a scan.
export function massCurveField(field, vmin, vmax) {
  const per = new Float64Array(MASS_BINS), span = (vmax - vmin) || 1;
  for (let i = 0; i < field.length; i++) {
    let b = (field[i] * MASS_BINS) | 0;
    if (b < 0) b = 0; else if (b >= MASS_BINS) b = MASS_BINS - 1;
    per[b] += vmin + ((b + 0.5) / MASS_BINS) * span;   // raw value, bin-centred
  }
  const above = new Float64Array(MASS_BINS + 1);
  for (let b = MASS_BINS - 1; b >= 0; b--) above[b] = above[b + 1] + per[b];
  return above;
}

// The same, straight off a wire grid, without materialising the float field.
export function massCurve(g) {
  const n = g.nx * g.ny * g.nz, per = new Float64Array(MASS_BINS);
  const span = (g.vmax - g.vmin) || 1;
  const T = g.f16 ? halfToFloatTable() : null;
  for (let i = 0; i < n; i++) {
    const v = T ? T[g.data[i]] : g.data[i] / 255;
    let b = (v * MASS_BINS) | 0;
    if (b < 0) b = 0; else if (b >= MASS_BINS) b = MASS_BINS - 1;
    per[b] += g.vmin + ((b + 0.5) / MASS_BINS) * span;
  }
  const above = new Float64Array(MASS_BINS + 1);
  for (let b = MASS_BINS - 1; b >= 0; b--) above[b] = above[b + 1] + per[b];
  return above;
}

// Mass above a normalized level, interpolated inside the bin so it is the exact inverse of
// levelForMass (round-tripping a level through the two must not move it).
export function massAbove(above, level) {
  const x = Math.max(0, Math.min(1, level)) * MASS_BINS;
  const b = Math.min(MASS_BINS - 1, Math.floor(x)), f = x - b;
  return above[b] + (above[b + 1] - above[b]) * f;
}

// The normalized level whose mass-above matches `target` — the client-side twin of
// meshing.level_for_mass. Mass falls as the level rises, so the search walks DOWN from the
// top for the first bin edge that still meets the target and interpolates inside that bin.
export function levelForMass(above, target) {
  if (!(target > 0)) return 1;
  if (above[0] <= target) return 0;              // the whole grid holds no more than this
  let b = MASS_BINS - 1;
  while (b >= 0 && above[b] < target) b--;
  const hi = above[b], lo = above[b + 1];
  const f = hi > lo ? (hi - target) / (hi - lo) : 0;   // 0 = the bin's low edge
  return Math.max(0, Math.min(1, (b + f) / MASS_BINS));
}

// ---------------------------------------------------------------------------
// growable typed-array buffers
// ---------------------------------------------------------------------------

function growF32(buf, need) {
  if (need <= buf.length) return buf;
  let cap = buf.length || 1024;
  while (cap < need) cap *= 2;
  const next = new Float32Array(cap);
  next.set(buf);
  return next;
}
function growU32(buf, need) {
  if (need <= buf.length) return buf;
  let cap = buf.length || 1024;
  while (cap < need) cap *= 2;
  const next = new Uint32Array(cap);
  next.set(buf);
  return next;
}

// ---------------------------------------------------------------------------
// marching tetrahedra
// ---------------------------------------------------------------------------

// The eight cube corners, as (dx, dy, dz) with the corner index's bits being x, y, z.
const CORNER = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];
// Six tetrahedra sharing the cube's 0-7 body diagonal, one per cube face. Every cube in the
// lattice uses the same diagonal, so two cubes sharing a face split it along the same
// diagonal and their triangles meet exactly — this is what makes the surface watertight
// across cube boundaries without any special-casing.
const TETS = [
  [0, 7, 1, 3], [0, 7, 3, 2], [0, 7, 2, 6],
  [0, 7, 6, 4], [0, 7, 4, 5], [0, 7, 5, 1],
];
// The seven edges a voxel OWNS: the three axis edges, the three diagonals of its three
// lower faces, and its body diagonal. Every edge of every tetrahedron in the lattice is one
// of these, owned by the voxel at the componentwise minimum of its two corners — which is
// how a vertex on a shared edge gets created once and indexed by both cubes.
const OWNED = [
  [1, 0, 0], [0, 1, 0], [0, 0, 1],
  [1, 1, 0], [1, 0, 1], [0, 1, 1],
  [1, 1, 1],
];
function ownedSlot(dx, dy, dz) {
  for (let k = 0; k < 7; k++) {
    const o = OWNED[k];
    if (o[0] === dx && o[1] === dy && o[2] === dz) return k;
  }
  return -1;
}
// Per (tetrahedron, edge-of-that-tetrahedron): which voxel offset owns it and which of the
// seven slots it is. Precomputed once — inside the loop this is a table lookup.
const TET_EDGES = TETS.map((tet) => {
  const pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
  return pairs.map(([x, y]) => {
    const a = CORNER[tet[x]], b = CORNER[tet[y]];
    const mi = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])];
    const d = [Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2])];
    return { ox: mi[0], oy: mi[1], oz: mi[2], slot: ownedSlot(d[0], d[1], d[2]) };
  });
});
// Which of the six pair-indices above connects tetrahedron corners (p, q).
const PAIR_OF = [[-1, 0, 1, 2], [0, -1, 3, 4], [1, 3, -1, 5], [2, 4, 5, -1]];

/**
 * Contour a scalar field at `level`.
 *
 * `field` is nx*ny*nz float samples, x fastest (the layout the grid payloads and the 3D
 * textures both use). `stride` samples every n-th voxel on every axis — a draft, in the same
 * cube as the full-resolution surface, exactly as the server's `size=` drafts are.
 *
 * Positions land in the model-space box [boxLo, boxHi], with sample j of the FULL grid at
 * lo + (j + 0.5) * (hi - lo) / n. Returns an indexed mesh with shared vertices and outward
 * (down-gradient) normals.
 */
export function marchingCubes(field, nx, ny, nz, level, opts = {}) {
  const stride = Math.max(1, opts.stride | 0 || 1);
  const boxLo = opts.boxLo || [-1, -1, -1];
  const boxHi = opts.boxHi || [1, 1, 1];
  // Sample grid: how many strided samples fit, and where sample I sits in model space.
  const mx = Math.floor((nx - 1) / stride) + 1;
  const my = Math.floor((ny - 1) / stride) + 1;
  const mz = Math.floor((nz - 1) / stride) + 1;
  const cell = [(boxHi[0] - boxLo[0]) / nx, (boxHi[1] - boxLo[1]) / ny, (boxHi[2] - boxLo[2]) / nz];
  const org = [boxLo[0] + 0.5 * cell[0], boxLo[1] + 0.5 * cell[1], boxLo[2] + 0.5 * cell[2]];
  const sc = [cell[0] * stride, cell[1] * stride, cell[2] * stride];

  let positions = new Float32Array(0), np = 0;      // np = vertex count
  let indices = new Uint32Array(0), ni = 0;         // ni = index count

  if (mx < 2 || my < 2 || mz < 2) {
    return { positions: new Float32Array(0), normals: new Float32Array(0), indices: new Uint32Array(0) };
  }

  // Vertex ids for the seven edges each voxel owns, over two z slices only: a cube at z
  // touches owners at z and z+1 and nothing further, so the whole lattice never has to be
  // resident. -1 means "no vertex on this edge yet".
  const planeSlots = mx * my * 7;
  const edgeId = new Int32Array(planeSlots * 2).fill(-1);
  // Flat offsets between the eight corners of a strided cube: the scan runs over every cube
  // in the volume, so it reads them by pointer arithmetic rather than through an index
  // helper — the difference is most of the scan's cost, and the scan is most of the cost of
  // contouring a map that is mostly empty.
  const nxy = nx * ny;
  const sx = stride, sy = stride * nx, sz = stride * nxy;

  const vv = new Float64Array(8);        // the current cube's corner values
  const tv = new Float64Array(4);        // the current tetrahedron's corner values
  const tri = new Int32Array(4);         // scratch: vertex ids of one polygon
  // Scratch for the per-tetrahedron classification. Allocating these inside the loop (the
  // obvious way to write it) put four short-lived arrays on the heap per crossing
  // tetrahedron, and the garbage collector then dominated the whole contour.
  const inC = new Int32Array(4), outC = new Int32Array(4);
  const pairA = new Int32Array(4), pairB = new Int32Array(4);
  const px = new Float64Array(4), py = new Float64Array(4), pz = new Float64Array(4);

  // Create (or reuse) the vertex on the edge between cube corners p and q of the tet.
  function edgeVertex(I, J, K, t, e) {
    const spec = TET_EDGES[t][e];
    const oz = K + spec.oz;
    const slot = (((oz & 1) * my + (J + spec.oy)) * mx + (I + spec.ox)) * 7 + spec.slot;
    const hit = edgeId[slot];
    if (hit >= 0) return hit;
    return -1 - slot;      // negative: "not yet, and here is where to store it"
  }

  for (let K = 0; K < mz - 1; K++) {
    // The slice that is about to be written for the first time holds ids from K-2; clear it.
    const clearBase = ((K + 1) & 1) * planeSlots;
    edgeId.fill(-1, clearBase, clearBase + planeSlots);
    if (K === 0) edgeId.fill(-1, 0, planeSlots);
    for (let J = 0; J < my - 1; J++) {
      for (let I = 0; I < mx - 1; I++) {
        const base = K * sz + J * sy + I * sx;
        const v0 = field[base], v1 = field[base + sx], v2 = field[base + sy], v3 = field[base + sx + sy];
        const v4 = field[base + sz], v5 = field[base + sx + sz], v6 = field[base + sy + sz], v7 = field[base + sx + sy + sz];
        // Occupancy pre-pass: a cube entirely on one side of the level cannot contain any
        // surface, and in a typical map that is well over 95% of them. Everything below —
        // six tetrahedra, their edges, the interpolations — is skipped for those.
        const below = (v0 < level) + (v1 < level) + (v2 < level) + (v3 < level)
          + (v4 < level) + (v5 < level) + (v6 < level) + (v7 < level);
        if (below === 0 || below === 8) continue;
        vv[0] = v0; vv[1] = v1; vv[2] = v2; vv[3] = v3;
        vv[4] = v4; vv[5] = v5; vv[6] = v6; vv[7] = v7;

        for (let t = 0; t < 6; t++) {
          const tet = TETS[t];
          let mask = 0;
          for (let c = 0; c < 4; c++) {
            const cc = tet[c];
            tv[c] = vv[cc];
            if (tv[c] >= level) mask |= 1 << c;      // "inside" = at or above the contour
          }
          if (mask === 0 || mask === 15) continue;

          // Which corners are in and which are out; the polygon spans the edges between the
          // two groups, which is 3 edges (1 vs 3) or 4 edges (2 vs 2).
          let nIn = 0, nOut = 0;
          for (let c = 0; c < 4; c++) { if (mask & (1 << c)) inC[nIn++] = c; else outC[nOut++] = c; }
          let nPts = 0;                               // 3 or 4
          for (let a = 0; a < nIn; a++) {
            for (let b = 0; b < nOut; b++) { pairA[nPts] = inC[a]; pairB[nPts] = outC[b]; nPts++; }
          }

          // Interpolate along each crossing edge, creating the vertex the first time any
          // tetrahedron in any cube asks for it.
          for (let m = 0; m < nPts; m++) {
            const a = pairA[m], b = pairB[m];
            const e = PAIR_OF[a][b];
            const got = edgeVertex(I, J, K, t, e);
            let id;
            if (got >= 0) {
              id = got;
              px[m] = positions[id * 3]; py[m] = positions[id * 3 + 1]; pz[m] = positions[id * 3 + 2];
            } else {
              const slot = -1 - got;
              const ca = CORNER[tet[a]], cb = CORNER[tet[b]];
              const va = tv[a], vb = tv[b];
              const denom = va - vb;
              const f = Math.abs(denom) > 1e-20 ? (va - level) / denom : 0.5;   // a -> b
              const gx = (I + ca[0]) + f * (cb[0] - ca[0]);
              const gy = (J + ca[1]) + f * (cb[1] - ca[1]);
              const gz = (K + ca[2]) + f * (cb[2] - ca[2]);
              const x = org[0] + sc[0] * gx, y = org[1] + sc[1] * gy, z = org[2] + sc[2] * gz;
              positions = growF32(positions, (np + 1) * 3);
              positions[np * 3] = x; positions[np * 3 + 1] = y; positions[np * 3 + 2] = z;
              id = np++;
              edgeId[slot] = id;
              px[m] = x; py[m] = y; pz[m] = z;
            }
            tri[m] = id;
          }

          // Orientation, decided per polygon from the field itself rather than from a table:
          // the normal must point DOWN the gradient (out of the high-density region), so it
          // must have a positive dot with (centroid of the outside corners - centroid of the
          // inside corners). Doing it this way makes the winding correct by construction for
          // every tetrahedron orientation and every case, which a table has to get right 256
          // times over.
          let ix = 0, iy = 0, iz = 0, ox = 0, oy = 0, oz = 0;
          for (let m = 0; m < nIn; m++) { const q = CORNER[tet[inC[m]]]; ix += q[0]; iy += q[1]; iz += q[2]; }
          for (let m = 0; m < nOut; m++) { const q = CORNER[tet[outC[m]]]; ox += q[0]; oy += q[1]; oz += q[2]; }
          const dx = (ox / nOut - ix / nIn) * sc[0];
          const dy = (oy / nOut - iy / nIn) * sc[1];
          const dz = (oz / nOut - iz / nIn) * sc[2];
          // 3 crossing edges -> one triangle; 4 -> a quad. The pair ordering above walks
          // in0xout0, in0xout1, in1xout0, in1xout1, so the quad's BOUNDARY order is 0,1,3,2 —
          // splitting it any other way would cut across the surface rather than along it.
          const quad = nPts === 4;
          const m0 = 0, m1 = 1, m2 = quad ? 3 : 2, m3 = 2;
          const a0 = tri[m0], a1 = tri[m1], a2 = tri[m2], a3 = quad ? tri[m3] : -1;
          // The winding test uses the first EMITTED triangle, not the first three crossing
          // points: for a quad those are not the same triangle.
          const e0x = px[m1] - px[m0], e0y = py[m1] - py[m0], e0z = pz[m1] - pz[m0];
          const e1x = px[m2] - px[m0], e1y = py[m2] - py[m0], e1z = pz[m2] - pz[m0];
          const cxx = e0y * e1z - e0z * e1y, cyy = e0z * e1x - e0x * e1z, czz = e0x * e1y - e0y * e1x;
          const flip = (cxx * dx + cyy * dy + czz * dz) < 0;
          indices = growU32(indices, ni + (quad ? 6 : 3));
          if (flip) {
            indices[ni++] = a0; indices[ni++] = a2; indices[ni++] = a1;
            if (quad) { indices[ni++] = a0; indices[ni++] = a3; indices[ni++] = a2; }
          } else {
            indices[ni++] = a0; indices[ni++] = a1; indices[ni++] = a2;
            if (quad) { indices[ni++] = a0; indices[ni++] = a2; indices[ni++] = a3; }
          }
        }
      }
    }
  }

  const pos = positions.subarray(0, np * 3).slice();
  const idx = indices.subarray(0, ni).slice();
  if (opts.smooth > 0) taubin(pos, idx, opts.smooth);
  return { positions: pos, normals: vertexNormals(pos, idx), indices: idx };
}

// Area-weighted per-vertex normals, exactly meshing._vertex_normals: the cross product of
// each face's two edges (its magnitude IS twice the area, so no explicit weight is needed)
// accumulated onto its three vertices, then normalized. Because the winding is outward, so
// are these.
export function vertexNormals(positions, indices) {
  const nv = positions.length / 3;
  const n = new Float64Array(nv * 3);
  for (let f = 0; f < indices.length; f += 3) {
    const a = indices[f] * 3, b = indices[f + 1] * 3, c = indices[f + 2] * 3;
    const e0x = positions[b] - positions[a], e0y = positions[b + 1] - positions[a + 1], e0z = positions[b + 2] - positions[a + 2];
    const e1x = positions[c] - positions[a], e1y = positions[c + 1] - positions[a + 1], e1z = positions[c + 2] - positions[a + 2];
    const fx = e0y * e1z - e0z * e1y, fy = e0z * e1x - e0x * e1z, fz = e0x * e1y - e0y * e1x;
    n[a] += fx; n[a + 1] += fy; n[a + 2] += fz;
    n[b] += fx; n[b + 1] += fy; n[b + 2] += fz;
    n[c] += fx; n[c + 1] += fy; n[c + 2] += fz;
  }
  const out = new Float32Array(nv * 3);
  for (let v = 0; v < nv; v++) {
    const x = n[v * 3], y = n[v * 3 + 1], z = n[v * 3 + 2];
    const l = Math.hypot(x, y, z) || 1;
    out[v * 3] = x / l; out[v * 3 + 1] = y / l; out[v * 3 + 2] = z / l;
  }
  return out;
}

// Taubin λ|μ smoothing, in place, mirroring meshing._taubin_smooth including its constants
// and its duplicate-edge weighting (an edge shared by two triangles is counted twice, both
// in the neighbour sum and in the degree, which is what the server's sparse matrix does).
export function taubin(positions, indices, iterations, lam = 0.5, mu = -0.53) {
  const it = iterations | 0;
  if (it <= 0 || !positions.length || !indices.length) return positions;
  const nv = positions.length / 3;
  // CSR-ish adjacency with multiplicity: count, then fill.
  const deg = new Int32Array(nv);
  const bump = (a, b) => { deg[a]++; deg[b]++; };
  for (let f = 0; f < indices.length; f += 3) {
    bump(indices[f], indices[f + 1]); bump(indices[f + 1], indices[f + 2]); bump(indices[f + 2], indices[f]);
  }
  const start = new Int32Array(nv + 1);
  for (let v = 0; v < nv; v++) start[v + 1] = start[v] + deg[v];
  const nbr = new Int32Array(start[nv]);
  const fill = start.slice(0, nv);
  const put = (a, b) => { nbr[fill[a]++] = b; nbr[fill[b]++] = a; };
  for (let f = 0; f < indices.length; f += 3) {
    put(indices[f], indices[f + 1]); put(indices[f + 1], indices[f + 2]); put(indices[f + 2], indices[f]);
  }
  const step = (w) => {
    const src = Float64Array.from(positions);
    for (let v = 0; v < nv; v++) {
      const s = start[v], e = start[v + 1], d = e - s;
      if (!d) continue;
      let sx = 0, sy = 0, sz = 0;
      for (let k = s; k < e; k++) { const q = nbr[k] * 3; sx += src[q]; sy += src[q + 1]; sz += src[q + 2]; }
      positions[v * 3] = src[v * 3] + w * (sx / d - src[v * 3]);
      positions[v * 3 + 1] = src[v * 3 + 1] + w * (sy / d - src[v * 3 + 1]);
      positions[v * 3 + 2] = src[v * 3 + 2] + w * (sz / d - src[v * 3 + 2]);
    }
  };
  for (let i = 0; i < it; i++) { step(lam); step(mu); }
  return positions;
}
