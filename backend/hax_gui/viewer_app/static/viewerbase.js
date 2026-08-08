// Request plumbing shared by the viewer frontends (app.js, mrc.js).

// Coalescing single-flight runner: fires immediately, and if more updates arrive while a
// request is in flight, re-runs once with the latest values when it returns. Gives
// continuous (server-bounded) updates without piling up requests.
//
// It must *wait* for the in-flight request, not abandon it. This used to abort the fetch
// on supersede, to avoid downloading a mesh that was already stale — but `/api/decode` is
// a sync route, so it runs in a threadpool task that cannot be cancelled: aborting freed
// the browser to fire the next request while the server kept grinding on the abandoned
// one. A drag therefore stacked up marching-cubes jobs that don't parallelise (8 at once
// measured ~5x slower *each*), so the further you dragged the slower it got. Waiting keeps
// the server at exactly one request at a time.
export function makeLive(fn) {
  let busy = false, pending = false, args = [];
  async function run() { busy = true; do { pending = false; try { await fn(...args); } catch { /* fn reports */ } } while (pending); busy = false; }
  return (...a) => { args = a; if (busy) pending = true; else run(); };
}

// Newest-request-wins guard for async loads that can overlap (switching the
// active map while a grid is still downloading): take a token before the first
// await, and bail if another request has taken one since.
//
//   const tok = guard.take();
//   const data = await fetchThing();
//   if (guard.stale(tok)) return;
export function latestWins() {
  let current = 0;
  return {
    take: () => ++current,
    stale: (tok) => tok !== current,
  };
}
