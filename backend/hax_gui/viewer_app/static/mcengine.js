// Main-thread client for the contouring workers: lifecycle, placement, request identity, and
// the two pieces of policy worth stating — newest wins, and the engine owns the memory.
//
// A contour drag issues a request per pointer move. Queuing them would make the surface lag
// further behind the slider the longer you drag (the failure mode viewerbase.makeLive was
// written to avoid on the server path). Here each request carries a channel — a model, or
// "morph", or one morph leg — and only the newest request on a channel can resolve; a reply
// that arrives after a newer request was issued is dropped on the floor. The worker still
// finishes the stale march, because a march is not cancellable, but nothing waits for it.
//
// There is a POOL of workers, not one. A march is a solid half-second of one core and a
// screenful of maps is ten of them, so one worker made "show ten maps" ten marches long no
// matter how many cores the machine had. Each field lives in exactly one worker — a field is
// 131 MB at a 320³ box, so replicating them to make every worker able to serve every request
// is not on the table — and the engine remembers which, which is what `placement` below is.
// One core is left for the page itself: the main thread still has to upload the meshes, and a
// pool that saturates every core just moves the freeze rather than removing it.
//
// Module workers are the only form used here (mccore.js is an ES module and is shared with
// the page and with node). Where they are unsupported, `available` is false and every caller
// falls back to the server path it already has.

// How much decoded field the whole pool may hold. Ten 320³ maps decode to 1.3 GB, which no
// browser tab should be asked to keep, so beyond this the least recently used field is
// dropped. What that costs is one ~100 ms decode per evicted map per pass over the scene, and
// only on scenes bigger than the budget — against a tab that would otherwise be killed.
const BUDGET_BYTES = 768 * 1024 * 1024;

export function createMcEngine() {
  const hw = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
  const want = Math.max(1, Math.min(4, hw - 1));
  const pool = [];                // { worker, dead }
  try {
    if (typeof Worker !== "undefined") {
      for (let i = 0; i < want; i++) {
        pool.push({ worker: new Worker(new URL("./mcworker.js", import.meta.url), { type: "module" }), dead: false });
      }
    }
  } catch { pool.length = 0; }

  let nextId = 1;
  const pending = new Map();      // reqId -> { resolve, channel, idx }
  const latest = new Map();       // channel -> newest reqId issued
  const placement = new Map();    // key -> index of the worker holding it
  const sizes = new Map();        // key -> decoded bytes, for the budget
  // Recency, oldest first: a Map preserves insertion order, so delete + set is "make newest".
  const recent = new Map();
  let heldBytes = 0;

  const alive = () => pool.some((w) => !w.dead);

  function forget(key) {
    placement.delete(key);
    recent.delete(key);
    if (sizes.has(key)) { heldBytes -= sizes.get(key); sizes.delete(key); }
  }
  function touch(key) {
    if (!recent.has(key)) return;
    recent.delete(key); recent.set(key, 1);
  }
  // The worker with the fewest bytes in it. Placement is by weight rather than by count
  // because a draft-sized grid and a native one differ by two orders of magnitude, and it is
  // the bytes that decide how long that worker's next march takes.
  function leastLoaded() {
    let best = -1, bestBytes = Infinity;
    for (let i = 0; i < pool.length; i++) {
      if (pool[i].dead) continue;
      let bytes = 0;
      for (const [k, idx] of placement) if (idx === i) bytes += sizes.get(k) || 0;
      if (bytes < bestBytes) { bestBytes = bytes; best = i; }
    }
    return best;
  }

  for (let i = 0; i < pool.length; i++) {
    const slot = pool[i];
    slot.worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.op === "stored") return;          // placement was recorded optimistically at store
      const p = pending.get(m.reqId);
      if (!p) return;
      pending.delete(m.reqId);
      // A miss means THIS worker does not have that field. Only forget it if this is where the
      // engine believed it was: a morph whose two maps were placed apart also misses, and
      // forgetting a field that is alive and well in another worker would strand 131 MB there
      // with nothing left to send it a `drop`.
      if (m.op === "miss") {
        if (placement.get(m.key) === i) forget(m.key);
        p.resolve({ miss: m.key });
        return;
      }
      if (m.op === "error") { p.resolve({ error: m.message }); return; }
      // Superseded: a newer request on this channel was issued while this one was marching.
      if (latest.get(p.channel) !== m.reqId) { p.resolve(null); return; }
      p.resolve({ positions: m.positions, normals: m.normals, indices: m.indices, level: m.level });
    };
    // A worker that threw takes its fields with it: forget them, so the next request for one
    // is placed somewhere else instead of waiting for a reply that will never come. Requests
    // already in flight there are failed rather than left hanging — a Prepare loop awaiting a
    // frame forever is a worse failure than one that reports an error.
    slot.worker.onerror = () => {
      slot.dead = true;
      for (const [k, idx] of [...placement]) if (idx === i) forget(k);
      for (const [reqId, p] of [...pending]) {
        if (p.idx !== i) continue;
        pending.delete(reqId);
        p.resolve({ error: "contouring worker failed" });
      }
    };
  }

  function send(msg, channel, idx) {
    if (idx < 0 || !pool[idx] || pool[idx].dead) return Promise.resolve({ error: "worker unavailable" });
    const reqId = nextId++;
    latest.set(channel, reqId);
    msg.reqId = reqId;
    return new Promise((resolve) => {
      pending.set(reqId, { resolve, channel, idx });
      pool[idx].worker.postMessage(msg);
    });
  }

  return {
    get available() { return alive(); },
    // Is this exact grid already decoded in the pool? Keys carry the grid's identity
    // (model, size, sample type), so a refetch at a different resolution is a different key
    // and cannot be confused with the old one.
    has: (key) => placement.has(key),
    // The grid payload is copied into the worker (structured clone of the typed array), which
    // is the price of the worker holding it across requests. ~65 MB once per grid.
    //
    // `opts.near` names another key this field must sit beside: a morph frame blends two
    // fields, so its two grids have to be in the SAME worker or the frame cannot be computed
    // at all. Without it the field goes wherever there is most room.
    store(key, grid, opts = {}) {
      if (!alive()) return;
      const bytes = grid.nx * grid.ny * grid.nz * 4;
      const beside = opts.near !== undefined ? placement.get(opts.near) : undefined;
      let idx = (beside !== undefined && !pool[beside].dead) ? beside
              : placement.has(key) ? placement.get(key)
              : leastLoaded();
      if (idx < 0 || pool[idx].dead) idx = leastLoaded();
      if (idx < 0) return;
      // Already exactly where it should be: nothing to send, just make it the newest.
      const at = placement.get(key);
      if (at === idx && sizes.get(key) === bytes) { touch(key); return; }
      // Moving a field (a morph leg wants it beside its partner): take it out of its old home
      // first, or the tab holds two copies of the same 131 MB with only one of them tracked.
      if (at !== undefined && at !== idx) {
        if (!pool[at].dead) pool[at].worker.postMessage({ op: "drop", key });
        forget(key);
      }
      // Make room. The field being stored and the one it must sit beside are never candidates:
      // evicting either of them is how a store turns into an infinite retry.
      for (const k of [...recent.keys()]) {
        if (heldBytes + bytes <= BUDGET_BYTES) break;
        if (k === key || k === opts.near) continue;
        const owner = placement.get(k);
        if (owner !== undefined && !pool[owner].dead) pool[owner].worker.postMessage({ op: "drop", key: k });
        forget(k);
      }
      forget(key);
      placement.set(key, idx);
      sizes.set(key, bytes);
      heldBytes += bytes;
      recent.set(key, 1);            // newest, so the budget evicts it last
      pool[idx].worker.postMessage({ op: "store", key, grid: {
        nx: grid.nx, ny: grid.ny, nz: grid.nz, vmin: grid.vmin, vmax: grid.vmax,
        boxLo: grid.boxLo, boxHi: grid.boxHi, f16: !!grid.f16, data: grid.data,
      } });
    },
    drop(key) {
      const idx = placement.get(key);
      if (idx !== undefined && pool[idx] && !pool[idx].dead) pool[idx].worker.postMessage({ op: "drop", key });
      forget(key);
    },
    // Resolves with the mesh, or null if superseded, or {miss} if the field is not where the
    // engine thought it was (the caller re-stores and asks again), or {error}.
    contour(key, { level, smooth = 0, stride = 1 }) {
      touch(key);
      const idx = placement.has(key) ? placement.get(key) : leastLoaded();
      return send({ op: "contour", key, level, smooth, stride }, key, idx);
    },
    // One morph frame: the blend of two resident fields, contoured. `massTarget` > 0 solves
    // the level for constant enclosed mass on the blended field itself, which is what the
    // server does per frame and what the GPU morph can only approximate.
    //
    // Both fields must be in one worker. When they are not — a leg whose two maps were placed
    // apart, or one that has been evicted — the request goes to whichever worker holds keyA
    // and comes back as a {miss} naming the absent key, and the caller re-stores that one with
    // `near` set so the pair ends up together and the retry succeeds.
    //
    // `channel` defaults to the single "morph" channel the scrubber uses, where newest-wins is
    // exactly what is wanted. A Prepare pass that contours several legs at once gives each leg
    // its own channel instead, so two legs in flight do not supersede each other.
    morphFrame({ keyA, keyB, t, scale = 1, off = 0, level = 0, massTarget = 0, smooth = 0, stride = 1,
                 channel = "morph" }) {
      touch(keyA); touch(keyB);
      const a = placement.get(keyA), b = placement.get(keyB);
      const idx = (a !== undefined && a === b) ? a
                : a !== undefined ? a
                : b !== undefined ? b
                : leastLoaded();
      return send({ op: "morphFrame", keyA, keyB, t, scale, off, level, massTarget, smooth, stride },
                  channel, idx);
    },
  };
}
