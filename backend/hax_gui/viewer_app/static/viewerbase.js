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

// Canvas video export. MediaRecorder's default bitrate (~2.5 Mbps) is fixed regardless of
// frame size, so a full-resolution canvas gets crushed into a blurry stream that looks far
// worse than the live view. Two fixes: a bitrate that grows with the pixel count (a
// generous 0.25 bit per pixel per frame — 1080p at 30 fps ≈ 15 Mbps), and, when the
// renderer supports it, drawing into a larger backing store for the duration so the movie
// carries the same detail a 2× snapshot does. Returns { stop() → Promise<Blob> }.
export const RECORD_SCALE = 2;
export function startCanvasRecording(target, fps, { scale = RECORD_SCALE } = {}) {
  if (target.setRenderScale) target.setRenderScale(scale);
  const canvas = target.canvas;
  const px = canvas ? canvas.width * canvas.height : 1920 * 1080;
  const videoBitsPerSecond = Math.min(80e6, Math.max(8e6, Math.round(0.25 * px * fps)));
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
  let rec;
  try {
    rec = new MediaRecorder(target.captureStream(fps), { mimeType: mime, videoBitsPerSecond });
  } catch (err) {
    if (target.setRenderScale) target.setRenderScale(null);
    throw err;
  }
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  rec.start();
  return {
    stop() {
      return new Promise((res) => {
        rec.onstop = () => {
          if (target.setRenderScale) target.setRenderScale(null);
          res(new Blob(chunks, { type: "video/webm" }));
        };
        rec.stop();
      });
    },
  };
}
