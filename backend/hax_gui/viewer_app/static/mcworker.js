// The contouring worker: a thin shell around mccore.js.
//
// It exists for one reason — marching a 320³ field takes the better part of a second, and on
// the main thread that is a second of frozen UI per contour, which is precisely the freeze
// this whole path is meant to remove. All the arithmetic lives in mccore.js so it can be
// tested under plain node; nothing here is worth testing except the plumbing.
//
// It holds decoded fields between requests. Decoding a 320³ half-precision grid into float32
// costs ~100 ms and 131 MB, and a contour drag issues one request per pointer move, so the
// field is stored once and every later contour is just the march. The LRU cap bounds that:
// four fields at a 320³ box is ~525 MB, which is the most a browser tab should be asked to
// hold. Beyond it the least recently used field is dropped and the next request for it comes
// back as a miss — the main thread still has the grid payload and simply re-stores it.
import { decodeField, marchingCubes, blendFields, massCurveField, levelForMass } from "./mccore.js";

const FIELD_CAP = 4;

const fields = new Map();   // key -> { field, nx, ny, nz, vmin, vmax, boxLo, boxHi }

function touch(key) {
  const rec = fields.get(key);
  if (!rec) return null;
  fields.delete(key); fields.set(key, rec);   // Map preserves insertion order: re-insert = MRU
  return rec;
}
function put(key, rec) {
  fields.delete(key);
  fields.set(key, rec);
  while (fields.size > FIELD_CAP) fields.delete(fields.keys().next().value);
}

// Contour one field and reply with the mesh, handing the buffers over rather than copying
// them: a full-resolution mesh is tens of megabytes and structured-cloning it would cost
// more than the march did.
function reply(reqId, mesh, extra = {}) {
  self.postMessage({ op: "mesh", reqId, positions: mesh.positions, normals: mesh.normals,
                     indices: mesh.indices, ...extra },
                   [mesh.positions.buffer, mesh.normals.buffer, mesh.indices.buffer]);
}

self.onmessage = (ev) => {
  const m = ev.data;
  try {
    if (m.op === "store") {
      put(m.key, {
        field: decodeField(m.grid), nx: m.grid.nx, ny: m.grid.ny, nz: m.grid.nz,
        vmin: m.grid.vmin, vmax: m.grid.vmax, boxLo: m.grid.boxLo, boxHi: m.grid.boxHi,
      });
      self.postMessage({ op: "stored", key: m.key });
      return;
    }
    if (m.op === "drop") { fields.delete(m.key); return; }

    if (m.op === "contour") {
      const rec = touch(m.key);
      if (!rec) { self.postMessage({ op: "miss", reqId: m.reqId, key: m.key }); return; }
      const mesh = marchingCubes(rec.field, rec.nx, rec.ny, rec.nz, m.level, {
        boxLo: rec.boxLo, boxHi: rec.boxHi, stride: m.stride, smooth: m.smooth,
      });
      reply(m.reqId, mesh);
      return;
    }

    if (m.op === "morphFrame") {
      const a = touch(m.keyA), b = touch(m.keyB);
      if (!a || !b) { self.postMessage({ op: "miss", reqId: m.reqId, key: a ? m.keyB : m.keyA }); return; }
      // The blend lives in the NEAR map's normalized space (mccore.blendFields), so the near
      // map's vmin/vmax are what turn it back into the raw units enclosed mass is measured in.
      const field = blendFields(a.field, b.field, m.t, m.scale, m.off);
      let level = m.level;
      if (m.massTarget > 0) {
        // Constant enclosed mass, solved on THIS frame's blended field — the thing the
        // interpolated-level approximation could only get right at the two endpoints, and the
        // reason the volume morph breathed before it had this.
        level = levelForMass(massCurveField(field, a.vmin, a.vmax), m.massTarget);
      }
      const mesh = marchingCubes(field, a.nx, a.ny, a.nz, level, {
        boxLo: a.boxLo, boxHi: a.boxHi, stride: m.stride, smooth: m.smooth,
      });
      reply(m.reqId, mesh, { level });
      return;
    }
  } catch (err) {
    self.postMessage({ op: "error", reqId: m.reqId, message: String(err && err.message || err) });
  }
};
