// Binary wire formats shared by the landscape viewer (app.js) and the map
// viewer (mrc.js). One parser per format, in one place — these used to be
// copy-pasted per app and had already started to drift.
//
// Every payload starts with an 8-byte prelude: "HXW1" + format id (u8) +
// format version (u8) + 2 reserved bytes (see wirefmt.py). Before this, a
// frontend/backend version mismatch — or an HTML error page fetched as
// arrayBuffer() — was parsed as geometry and rendered as garbage.

export const FMT = { mesh: 1, grid: 2, f32matrix: 3, i32vector: 4, f32vector: 5, cmesh: 6 };
const MAGIC = 0x48585731; // "HXW1"

export function unwrap(buf, fmt) {
  const dv = new DataView(buf);
  if (buf.byteLength < 8 || dv.getUint32(0, false) !== MAGIC) {
    throw new Error("unrecognized binary payload — the viewer page and its backend "
                    + "disagree on the wire format (reload the page / restart the viewer)");
  }
  if (dv.getUint8(4) !== fmt) throw new Error(`unexpected payload format ${dv.getUint8(4)} (wanted ${fmt})`);
  return buf.slice(8);
}

// int32 nverts, nfaces | float32 level, mass, vmax | positions, normals, indices.
// `mass` is the enclosed density above the contour: the morph holds it constant
// across a series, so it has to come back out of the header.
export function parseMesh(wireBuf) {
  const buf = unwrap(wireBuf, FMT.mesh);
  const dv = new DataView(buf);
  const nverts = dv.getInt32(0, true), nfaces = dv.getInt32(4, true);
  const level = dv.getFloat32(8, true), mass = dv.getFloat32(12, true);
  let off = 20;
  const positions = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const normals = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const indices = new Uint32Array(buf, off, nfaces * 3);
  return { positions, normals, indices, level, mass };
}

// The MESH layout plus `prange` (the ± range of the diverging colour ramp) in the
// header and a uint8 RGB triple per vertex after the indices — a surface painted by
// another map sampled on it (see /api/paint).
export function parseColorMesh(wireBuf) {
  const buf = unwrap(wireBuf, FMT.cmesh);
  const dv = new DataView(buf);
  const nverts = dv.getInt32(0, true), nfaces = dv.getInt32(4, true);
  const level = dv.getFloat32(8, true), mass = dv.getFloat32(12, true);
  const prange = dv.getFloat32(20, true);
  let off = 24;
  const positions = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const normals = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const indices = new Uint32Array(buf, off, nfaces * 3); off += nfaces * 12;
  const colors = new Uint8Array(buf, off, nverts * 3);
  return { positions, normals, indices, colors, level, mass, prange };
}

// int32 nx, ny, nz | float32 vmin, vmax, mean, std | float32 boxLo[3], boxHi[3] |
// uint8 data[nx*ny*nz].
export function parseGrid(wireBuf) {
  const buf = unwrap(wireBuf, FMT.grid);
  const dv = new DataView(buf);
  const nx = dv.getInt32(0, true), ny = dv.getInt32(4, true), nz = dv.getInt32(8, true);
  const vmin = dv.getFloat32(12, true), vmax = dv.getFloat32(16, true), mean = dv.getFloat32(20, true), std = dv.getFloat32(24, true);
  const boxLo = [dv.getFloat32(28, true), dv.getFloat32(32, true), dv.getFloat32(36, true)];
  const boxHi = [dv.getFloat32(40, true), dv.getFloat32(44, true), dv.getFloat32(48, true)];
  const data = new Uint8Array(buf, 52, nx * ny * nz);
  return { nx, ny, nz, vmin, vmax, mean, std, boxLo, boxHi, data };
}
