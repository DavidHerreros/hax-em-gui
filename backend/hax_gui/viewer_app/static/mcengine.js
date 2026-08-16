// Main-thread client for the contouring worker: lifecycle, request identity, and the one
// piece of policy worth stating — newest wins.
//
// A contour drag issues a request per pointer move. Queuing them would make the surface lag
// further behind the slider the longer you drag (the failure mode viewerbase.makeLive was
// written to avoid on the server path). Here each request carries a channel — a model, or
// "morph" — and only the newest request on a channel can resolve; a reply that arrives after
// a newer request was issued is dropped on the floor. The worker still finishes the stale
// march, because a march is not cancellable, but nothing waits for it.
//
// Module workers are the only form used here (mccore.js is an ES module and is shared with
// the page and with node). Where they are unsupported, `available` is false and every caller
// falls back to the server path it already has.
export function createMcEngine() {
  let worker = null, available = false;
  try {
    if (typeof Worker !== "undefined") {
      worker = new Worker(new URL("./mcworker.js", import.meta.url), { type: "module" });
      available = true;
    }
  } catch { worker = null; available = false; }

  let nextId = 1;
  const pending = new Map();      // reqId -> { resolve, channel }
  const latest = new Map();       // channel -> newest reqId issued
  const stored = new Set();       // keys the worker is believed to hold

  if (worker) {
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.op === "stored") { stored.add(m.key); return; }
      const p = pending.get(m.reqId);
      if (!p) return;
      pending.delete(m.reqId);
      if (m.op === "miss") { stored.delete(m.key); p.resolve({ miss: m.key }); return; }
      if (m.op === "error") { p.resolve({ error: m.message }); return; }
      // Superseded: a newer request on this channel was issued while this one was marching.
      if (latest.get(p.channel) !== m.reqId) { p.resolve(null); return; }
      p.resolve({ positions: m.positions, normals: m.normals, indices: m.indices, level: m.level });
    };
    worker.onerror = () => { available = false; };
  }

  function send(msg, channel) {
    if (!worker) return Promise.resolve({ error: "worker unavailable" });
    const reqId = nextId++;
    latest.set(channel, reqId);
    msg.reqId = reqId;
    return new Promise((resolve) => { pending.set(reqId, { resolve, channel }); worker.postMessage(msg); });
  }

  return {
    get available() { return available; },
    // Is this exact grid already decoded in the worker? Keys carry the grid's identity
    // (model, size, sample type), so a refetch at a different resolution is a different key
    // and cannot be confused with the old one.
    has: (key) => stored.has(key),
    // The grid payload is copied into the worker (structured clone of the typed array), which
    // is the price of the worker holding it across requests. ~65 MB once per grid.
    store(key, grid) {
      if (!worker) return;
      stored.add(key);   // optimistic: a miss reply un-sets it
      worker.postMessage({ op: "store", key, grid: {
        nx: grid.nx, ny: grid.ny, nz: grid.nz, vmin: grid.vmin, vmax: grid.vmax,
        boxLo: grid.boxLo, boxHi: grid.boxHi, f16: !!grid.f16, data: grid.data,
      } });
    },
    drop(key) { stored.delete(key); if (worker) worker.postMessage({ op: "drop", key }); },
    // Resolves with the mesh, or null if superseded, or {miss} if the worker's LRU has
    // dropped the field (the caller re-stores and asks again), or {error}.
    contour(key, { level, smooth = 0, stride = 1 }) {
      return send({ op: "contour", key, level, smooth, stride }, key);
    },
    // One morph frame: the blend of two resident fields, contoured. `massTarget` > 0 solves
    // the level for constant enclosed mass on the blended field itself, which is what the
    // server does per frame and what the GPU morph can only approximate.
    morphFrame({ keyA, keyB, t, scale = 1, off = 0, level = 0, massTarget = 0, smooth = 0, stride = 1 }) {
      return send({ op: "morphFrame", keyA, keyB, t, scale, off, level, massTarget, smooth, stride }, "morph");
    },
  };
}
