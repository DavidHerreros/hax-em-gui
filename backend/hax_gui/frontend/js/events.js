// Global job-event hub: one /ws/events socket per tab, fanned out to listeners.
//
// The server pushes a message per job status transition (see hax_gui/events.py).
// Views keep their pollers — that is the correctness fallback when the socket is
// down — but kick them on receipt, so a finished run shows up immediately
// instead of on the next 3–5 s tick. The socket is opened lazily on the first
// listener and reconnects forever (subscribeWS backoff).
import { subscribeWS } from "./api.js";

const subs = new Set();
let started = false;

function start() {
  if (started) return;
  started = true;
  subscribeWS("/ws/events", (m) => {
    for (const fn of [...subs]) {
      try { fn(m); } catch { /* one bad listener must not break the rest */ }
    }
  }, { shouldReconnect: () => true });
}

// Register a listener; returns an unsubscribe function.
export function onJobEvent(fn) {
  start();
  subs.add(fn);
  return () => subs.delete(fn);
}
