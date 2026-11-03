// The contouring worker: a thin shell around mccore.js.
//
// It exists for one reason — marching a 320³ field takes the better part of a second, and on
// the main thread that is a second of frozen UI per contour, which is precisely the freeze
// this whole path is meant to remove. All the arithmetic lives in mccore.js so it can be
// tested under plain node; nothing here is worth testing except the plumbing.
//
// Several copies of this module run at once (mcengine.js starts a small pool of them), so a
// screenful of maps contours on several threads instead of queueing on one. Each copy is
// completely independent: it holds whichever fields the engine placed in it and knows nothing
// about its siblings.
//
// It holds decoded fields between requests. Decoding a 320³ half-precision grid into float32
// costs ~100 ms and 131 MB, and a contour drag issues one request per pointer move, so the
// field is stored once and every later contour is just the march. There is deliberately NO cap
// here: how much a tab may hold is one question about the whole pool, not one per thread, and
// it is the engine that has the whole picture — a worker that evicted on its own count would
// throw away a field the engine still believes is resident. Instead the engine sends `drop`,
// and the `miss` reply below stays as the safety net: if a request names a field this worker
// does not have, the main thread still holds the grid payload and simply re-stores it.
//
// Every stored field carries its block min/max table (mccore.blockMinMax), built once at store
// time and handed to every march of that field: it is what lets the scan dismiss whole 8³
// neighbourhoods of vacuum, which is most of a cryoEM map.
import { decodeField, blockMinMax, marchingCubes, blendFields, massCurveField, levelForMass } from "./mccore.js";

const fields = new Map();   // key -> { field, blocks, nx, ny, nz, vmin, vmax, boxLo, boxHi }

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
      const { field, blocks } = decodeField(m.grid);
      fields.set(m.key, {
        field, blocks, nx: m.grid.nx, ny: m.grid.ny, nz: m.grid.nz,
        vmin: m.grid.vmin, vmax: m.grid.vmax, boxLo: m.grid.boxLo, boxHi: m.grid.boxHi,
      });
      self.postMessage({ op: "stored", key: m.key });
      return;
    }
    if (m.op === "drop") { fields.delete(m.key); return; }

    if (m.op === "contour") {
      const rec = fields.get(m.key);
      if (!rec) { self.postMessage({ op: "miss", reqId: m.reqId, key: m.key }); return; }
      const mesh = marchingCubes(rec.field, rec.nx, rec.ny, rec.nz, m.level, {
        boxLo: rec.boxLo, boxHi: rec.boxHi, stride: m.stride, smooth: m.smooth, blocks: rec.blocks,
      });
      reply(m.reqId, mesh);
      return;
    }

    if (m.op === "morphFrame") {
      const a = fields.get(m.keyA), b = fields.get(m.keyB);
      if (!a || !b) { self.postMessage({ op: "miss", reqId: m.reqId, key: a ? m.keyB : m.keyA }); return; }
      // The blend lives in the NEAR map's normalized space (mccore.blendFields), so the near
      // map's vmin/vmax are what turn it back into the raw units enclosed mass is measured in.
      const field = blendFields(a.field, b.field, m.t, m.scale, m.off);
      // A frame's field is neither map's, so neither map's block table describes it: the blend
      // gets its own. That is one extra pass over the samples against the several hundred
      // milliseconds of empty-space scanning it saves on the march that follows.
      const blocks = blockMinMax(field, a.nx, a.ny, a.nz);
      let level = m.level;
      if (m.massTarget > 0) {
        // Constant enclosed mass, solved on THIS frame's blended field — the thing the
        // interpolated-level approximation could only get right at the two endpoints, and the
        // reason the volume morph breathed before it had this.
        level = levelForMass(massCurveField(field, a.vmin, a.vmax), m.massTarget);
      }
      const mesh = marchingCubes(field, a.nx, a.ny, a.nz, level, {
        boxLo: a.boxLo, boxHi: a.boxHi, stride: m.stride, smooth: m.smooth, blocks,
      });
      reply(m.reqId, mesh, { level });
      return;
    }
  } catch (err) {
    self.postMessage({ op: "error", reqId: m.reqId, message: String(err && err.message || err) });
  }
};
