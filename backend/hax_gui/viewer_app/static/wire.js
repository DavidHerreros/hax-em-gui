// Binary wire formats shared by the landscape viewer (app.js) and the map
// viewer (mrc.js). One parser per format, in one place — these used to be
// copy-pasted per app and had already started to drift.

// int32 nverts, nfaces | float32 level, mass, vmax | positions, normals, indices.
// `mass` is the enclosed density above the contour: the morph holds it constant
// across a series, so it has to come back out of the header.
export function parseMesh(buf) {
  const dv = new DataView(buf);
  const nverts = dv.getInt32(0, true), nfaces = dv.getInt32(4, true);
  const level = dv.getFloat32(8, true), mass = dv.getFloat32(12, true);
  let off = 20;
  const positions = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const normals = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const indices = new Uint32Array(buf, off, nfaces * 3);
  return { positions, normals, indices, level, mass };
}

// int32 nx, ny, nz | float32 vmin, vmax, mean, std | float32 boxLo[3], boxHi[3] |
// uint8 data[nx*ny*nz].
export function parseGrid(buf) {
  const dv = new DataView(buf);
  const nx = dv.getInt32(0, true), ny = dv.getInt32(4, true), nz = dv.getInt32(8, true);
  const vmin = dv.getFloat32(12, true), vmax = dv.getFloat32(16, true), mean = dv.getFloat32(20, true), std = dv.getFloat32(24, true);
  const boxLo = [dv.getFloat32(28, true), dv.getFloat32(32, true), dv.getFloat32(36, true)];
  const boxHi = [dv.getFloat32(40, true), dv.getFloat32(44, true), dv.getFloat32(48, true)];
  const data = new Uint8Array(buf, 52, nx * ny * nz);
  return { nx, ny, nz, vmin, vmax, mean, std, boxLo, boxHi, data };
}
