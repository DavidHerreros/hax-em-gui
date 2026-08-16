// Numeric tests for mccore.js. Self-contained, no framework, no dependencies:
//
//     node mccore.test.mjs
//
// The claims worth testing about a contouring routine are geometric, not textual: every
// vertex must actually sit on the isosurface, the mesh must be closed (a hole is how a bad
// case table shows up, and it would otherwise only be noticed in a finished figure), and the
// normals must point out of the high-density region — the surface is lit by that convention,
// so getting it wrong turns a map inside out. Two topologies, because a sphere alone would
// not exercise the cases with more than one crossing per tetrahedron.
import { marchingCubes, taubin, vertexNormals, massCurveField, massAbove, levelForMass, blendFields,
  blendedMassCurve, massCurve } from "./mccore.js";

let failures = 0;
function check(name, ok, detail = "") {
  const tag = ok ? "  ok  " : "  FAIL";
  console.log(`${tag}  ${name}${detail ? "   (" + detail + ")" : ""}`);
  if (!ok) failures++;
}

const N = 64;
const BOX_LO = [-1, -1, -1], BOX_HI = [1, 1, 1];
// Sample j sits at lo + (j + 0.5) * (hi - lo) / n — the mapping mccore documents.
const coord = (j) => BOX_LO[0] + (j + 0.5) * (BOX_HI[0] - BOX_LO[0]) / N;

function buildField(fn) {
  const f = new Float32Array(N * N * N);
  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) f[(k * N + j) * N + i] = fn(coord(i), coord(j), coord(k));
    }
  }
  return f;
}
// Positive inside, as a density map is: high in the middle of the object.
const sphere = buildField((x, y, z) => 0.6 - Math.hypot(x, y, z));
const torus = buildField((x, y, z) => 0.22 - Math.hypot(Math.hypot(x, y) - 0.55, z));

// Trilinear sample of the field at a model-space point, for checking that vertices really
// are on the isosurface (the inverse of the mapping above).
function sampleAt(field, x, y, z) {
  const gx = (x - BOX_LO[0]) * N / (BOX_HI[0] - BOX_LO[0]) - 0.5;
  const gy = (y - BOX_LO[1]) * N / (BOX_HI[1] - BOX_LO[1]) - 0.5;
  const gz = (z - BOX_LO[2]) * N / (BOX_HI[2] - BOX_LO[2]) - 0.5;
  const cl = (v) => Math.max(0, Math.min(N - 1, v));
  const i0 = Math.floor(gx), j0 = Math.floor(gy), k0 = Math.floor(gz);
  const fx = gx - i0, fy = gy - j0, fz = gz - k0;
  const at = (i, j, k) => field[(cl(k) * N + cl(j)) * N + cl(i)];
  let s = 0;
  for (let dk = 0; dk < 2; dk++) {
    for (let dj = 0; dj < 2; dj++) {
      for (let di = 0; di < 2; di++) {
        const w = (di ? fx : 1 - fx) * (dj ? fy : 1 - fy) * (dk ? fz : 1 - fz);
        s += w * at(i0 + di, j0 + dj, k0 + dk);
      }
    }
  }
  return s;
}

// Every edge of a closed triangle mesh belongs to exactly two triangles.
function openEdges(indices) {
  const seen = new Map();
  for (let f = 0; f < indices.length; f += 3) {
    for (let e = 0; e < 3; e++) {
      const a = indices[f + e], b = indices[f + (e + 1) % 3];
      const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
      seen.set(key, (seen.get(key) || 0) + 1);
    }
  }
  let bad = 0;
  for (const c of seen.values()) if (c !== 2) bad++;
  return bad;
}

// The field's largest change between neighbouring samples: for a signed-distance field this
// is one voxel expressed in field units, which is what turns a residual into a distance.
function maxAdjacentDelta(field) {
  let m = 0;
  for (let k = 1; k < N - 1; k += 3) {
    for (let j = 1; j < N - 1; j += 3) {
      for (let i = 1; i < N - 1; i++) {
        m = Math.max(m, Math.abs(field[(k * N + j) * N + i] - field[(k * N + j) * N + i - 1]));
      }
    }
  }
  return m || 1;
}

function runTopology(label, field, level, center) {
  console.log(`\n--- ${label} (${N}^3, level ${level}) ---`);
  const t0 = process.hrtime.bigint();
  const mesh = marchingCubes(field, N, N, N, level, { boxLo: BOX_LO, boxHi: BOX_HI });
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  const nv = mesh.positions.length / 3, nf = mesh.indices.length / 3;
  console.log(`      ${nv} vertices, ${nf} triangles, ${ms.toFixed(1)} ms`);
  check("produces triangles", nf > 0, `${nf}`);

  let worstVal = 0, outside = 0;
  for (let v = 0; v < nv; v++) {
    const x = mesh.positions[v * 3], y = mesh.positions[v * 3 + 1], z = mesh.positions[v * 3 + 2];
    worstVal = Math.max(worstVal, Math.abs(sampleAt(field, x, y, z) - level));
    if (x < BOX_LO[0] || x > BOX_HI[0] || y < BOX_LO[1] || y > BOX_HI[1] || z < BOX_LO[2] || z > BOX_HI[2]) outside++;
  }
  // A vertex sits EXACTLY on the linear interpolation along its edge, which is the algorithm's
  // contract; measured against the trilinear reconstruction it differs by the second-order
  // term, so the honest claim is a distance, not a field value. Both test fields are signed
  // distances, so dividing the residual by the field's change across one voxel converts it
  // straight into voxels of positional error.
  const perVoxel = maxAdjacentDelta(field);
  check("every vertex is within 0.1 voxel of the true isosurface",
    worstVal / perVoxel < 0.1,
    `residual ${worstVal.toExponential(2)} = ${(worstVal / perVoxel).toFixed(3)} voxels`);
  check("every vertex is inside the box", outside === 0, `${outside} outside`);

  let worstLen = 0, inward = 0;
  for (let v = 0; v < nv; v++) {
    const nx = mesh.normals[v * 3], ny = mesh.normals[v * 3 + 1], nz = mesh.normals[v * 3 + 2];
    worstLen = Math.max(worstLen, Math.abs(Math.hypot(nx, ny, nz) - 1));
    const rx = mesh.positions[v * 3] - center[0], ry = mesh.positions[v * 3 + 1] - center[1], rz = mesh.positions[v * 3 + 2] - center[2];
    if (nx * rx + ny * ry + nz * rz <= 0) inward++;
  }
  check("normals are unit length (<1e-4)", worstLen < 1e-4, `max |len - 1| = ${worstLen.toExponential(2)}`);
  return { mesh, nv, nf, inward, ms };
}

// ---- sphere: outwardness is unambiguous, so it also pins the winding convention ----
const sph = runTopology("sphere", sphere, 0, [0, 0, 0]);
check("normals point outward (away from the dense centre)", sph.inward === 0, `${sph.inward} of ${sph.nv} inward`);
check("mesh is closed (every edge in exactly 2 triangles)", openEdges(sph.mesh.indices) === 0,
  `${openEdges(sph.mesh.indices)} bad edges`);

// ---- torus: a second topology, with genus 1 and saddle cases ----
const tor = runTopology("torus", torus, 0, [0, 0, 0]);
check("mesh is closed (every edge in exactly 2 triangles)", openEdges(tor.mesh.indices) === 0,
  `${openEdges(tor.mesh.indices)} bad edges`);
// On a torus "outward" is only well defined against the tube's own centre line.
{
  let inward = 0;
  for (let v = 0; v < tor.nv; v++) {
    const x = tor.mesh.positions[v * 3], y = tor.mesh.positions[v * 3 + 1], z = tor.mesh.positions[v * 3 + 2];
    const r = Math.hypot(x, y) || 1e-9;
    const cx = 0.55 * x / r, cy = 0.55 * y / r;      // nearest point on the tube's centre line
    const d = [x - cx, y - cy, z];
    if (tor.mesh.normals[v * 3] * d[0] + tor.mesh.normals[v * 3 + 1] * d[1] + tor.mesh.normals[v * 3 + 2] * d[2] <= 0) inward++;
  }
  check("normals point out of the tube", inward === 0, `${inward} of ${tor.nv} inward`);
}

// ---- taubin ----
{
  console.log("\n--- taubin (6 iterations, sphere) ---");
  const pos = sph.mesh.positions.slice();
  taubin(pos, sph.mesh.indices, 6);
  let maxMove = 0, sum = 0;
  for (let v = 0; v < sph.nv; v++) {
    const d = Math.hypot(pos[v * 3] - sph.mesh.positions[v * 3],
      pos[v * 3 + 1] - sph.mesh.positions[v * 3 + 1],
      pos[v * 3 + 2] - sph.mesh.positions[v * 3 + 2]);
    maxMove = Math.max(maxMove, d); sum += d;
  }
  const cell = (BOX_HI[0] - BOX_LO[0]) / N;
  console.log(`      mean move ${(sum / sph.nv / cell).toFixed(3)} voxels, max ${(maxMove / cell).toFixed(3)} voxels`);
  check("smoothing moves vertices, but by less than a voxel", maxMove > 0 && maxMove < cell,
    `max = ${(maxMove / cell).toFixed(3)} voxels`);
  check("smoothing keeps the mesh closed", openEdges(sph.mesh.indices) === 0);
  // The radius must survive: Taubin's whole point over plain Laplacian is not shrinking.
  let r0 = 0, r1 = 0;
  for (let v = 0; v < sph.nv; v++) {
    r0 += Math.hypot(sph.mesh.positions[v * 3], sph.mesh.positions[v * 3 + 1], sph.mesh.positions[v * 3 + 2]);
    r1 += Math.hypot(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]);
  }
  r0 /= sph.nv; r1 /= sph.nv;
  check("smoothing does not shrink the surface (<0.5%)", Math.abs(r1 - r0) / r0 < 0.005,
    `radius ${r0.toFixed(4)} -> ${r1.toFixed(4)}`);
  const nrm = vertexNormals(pos, sph.mesh.indices);
  let worst = 0;
  for (let v = 0; v < sph.nv; v++) worst = Math.max(worst, Math.abs(Math.hypot(nrm[v * 3], nrm[v * 3 + 1], nrm[v * 3 + 2]) - 1));
  check("normals recomputed after smoothing are unit length", worst < 1e-4);
}

// ---- stride (drafts must land in the same cube) ----
{
  console.log("\n--- stride 2 draft vs full ---");
  const draft = marchingCubes(sphere, N, N, N, 0, { boxLo: BOX_LO, boxHi: BOX_HI, stride: 2 });
  const nv = draft.positions.length / 3;
  let worst = 0;
  for (let v = 0; v < nv; v++) {
    const r = Math.hypot(draft.positions[v * 3], draft.positions[v * 3 + 1], draft.positions[v * 3 + 2]);
    worst = Math.max(worst, Math.abs(r - 0.6));
  }
  console.log(`      ${nv} vertices, max radius error ${worst.toFixed(4)} (cell = ${(2 / N).toFixed(4)})`);
  check("a strided draft sits on the same sphere (within a strided cell)", worst < 2 * (2 / N),
    `${worst.toFixed(4)} < ${(4 / N).toFixed(4)}`);
  check("draft is closed", openEdges(draft.indices) === 0);
}

// ---- mass arithmetic ----
{
  console.log("\n--- mass / level round trip ---");
  const f = new Float32Array(4096);
  for (let i = 0; i < f.length; i++) f[i] = i / f.length;
  const above = massCurveField(f, 0, 1);
  const lvl = 0.37;
  const back = levelForMass(above, massAbove(above, lvl));
  check("levelForMass inverts massAbove (<2e-3)", Math.abs(back - lvl) < 2e-3, `${lvl} -> ${back.toFixed(5)}`);
  const b = blendFields(new Float32Array([0, 0.5, 1]), new Float32Array([1, 1, 1]), 0.25, 2, -0.5);
  check("blendFields applies the A-space rescale", Math.abs(b[0] - 0.375) < 1e-6, `${b[0]}`);
}

// ---- the blended mass curve, and what subsampling costs it ----
//
// The constant-mass morph solves its level on this curve, so the question that matters is
// whether a SUBSAMPLED curve gives the same LEVEL as the full one, not whether the two curves
// match. Two synthetic uint8 grids with different ranges, so the A-space rescale is exercised
// too, and one narrow feature in A so subsampling has something to miss.
//
// The error is set by the number of samples, not by the stride: at 128^3 a stride of 4 leaves
// 32768 of them, where a real 320^3 map leaves 512000 — sixteen times as many — so the figure
// printed here is a pessimistic stand-in for what the viewer actually does.
{
  console.log("\n--- blended mass curve (128^3 pair, subsampled) ---");
  const M = 128, np = M * M * M;
  const mk = (f, vmin, vmax) => {
    const data = new Uint8Array(np);
    for (let k = 0; k < M; k++) {
      for (let j = 0; j < M; j++) {
        for (let i = 0; i < M; i++) {
          const x = (i / M) * 2 - 1, y = (j / M) * 2 - 1, z = (k / M) * 2 - 1;
          data[(k * M + j) * M + i] = Math.max(0, Math.min(255, Math.round(255 * f(x, y, z))));
        }
      }
    }
    return { nx: M, ny: M, nz: M, data, f16: false, vmin, vmax };
  };
  const gA = mk((x, y, z) => Math.exp(-4 * (x * x + y * y + z * z))
                             + 0.3 * Math.exp(-30 * ((x - 0.4) ** 2 + y * y + z * z)), -0.2, 1.8);
  const gB = mk((x, y, z) => Math.exp(-4 * ((x - 0.3) ** 2 + y * y + z * z)), 0.1, 2.4);
  const spanA = gA.vmax - gA.vmin;
  const scale = (gB.vmax - gB.vmin) / spanA, off = (gB.vmin - gA.vmin) / spanA;
  const refMass = massAbove(massCurve(gA), 0.5);
  const ts = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];
  for (const stride of [4, 2]) {
    let worst = 0;
    for (const t of ts) {
      const full = levelForMass(blendedMassCurve(gA, gB, t, scale, off, 1), refMass);
      const sub = levelForMass(blendedMassCurve(gA, gB, t, scale, off, stride), refMass);
      worst = Math.max(worst, Math.abs(full - sub));
    }
    console.log(`      stride ${stride} (${Math.round(np / stride ** 3)} samples): worst level error ${worst.toExponential(2)}`);
    check(`stride ${stride} solves the same level as the full curve (<5e-3 of the range)`,
      worst < 5e-3, `${worst.toExponential(2)}`);
  }
  // The endpoints must be the maps' own curves, or a leg boundary would step.
  const at0 = blendedMassCurve(gA, gB, 0, scale, off, 1), plain = massCurve(gA);
  let d = 0;
  for (let b = 0; b < plain.length; b += 64) d = Math.max(d, Math.abs(at0[b] - plain[b]) / (plain[0] || 1));
  check("at t = 0 the blended curve is map A's own curve", d < 1e-9, `max relative difference ${d.toExponential(2)}`);
}

// ---- timing, split into the two things that actually scale differently ----
//
// The scan (8 samples + the occupancy test per cube) is proportional to the VOLUME; the rest
// — tetrahedra, interpolation, vertex creation — is proportional to the number of cubes the
// surface passes through, which is an AREA. Measuring them apart is what makes an
// extrapolation to a 320³ box mean anything: a level nothing crosses times the scan alone.
console.log(`\n--- timing ---`);
{
  const t0 = process.hrtime.bigint();
  marchingCubes(sphere, N, N, N, 10, { boxLo: BOX_LO, boxHi: BOX_HI });   // no crossings at all
  const scan = Number(process.hrtime.bigint() - t0) / 1e6;
  const surf = sph.ms - scan;
  const vox = N ** 3, big = 320 ** 3;
  console.log(`      ${N}^3 sphere: ${sph.ms.toFixed(1)} ms total = ${scan.toFixed(1)} ms scan + ${surf.toFixed(1)} ms surface`);
  console.log(`      scan: ${(scan / vox * 1e6).toFixed(1)} ns/voxel -> ${(scan * big / vox).toFixed(0)} ms at 320^3`);
  console.log(`      surface: ${(surf / sph.nf * 1e6).toFixed(0)} ns/triangle`);
  console.log(`      a 320^3 map with ~1.5M triangles: ~${(scan * big / vox + surf / sph.nf * 1.5e6).toFixed(0)} ms`);
  console.log(`      (the server's full pass is ~750 ms + ~21 MB over the network for the same map)`);
}

console.log(`\n${failures ? failures + " FAILURE(S)" : "all checks passed"}`);
process.exit(failures ? 1 : 0);
