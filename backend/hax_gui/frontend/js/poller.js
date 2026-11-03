// Shared polling loop. Every polling view used to hand-roll setInterval and
// re-decide the same three questions — none had an in-flight guard (slow
// responses stacked into poll-amplification), error handling varied from
// "retry" to "stop forever", and only one view skipped hidden tabs. This
// helper answers them once:
//
//   - one call in flight at a time (a tick is skipped, never queued);
//   - `fn`'s rejections never stop the loop — the view owns its error UI and
//     the next tick retries;
//   - hidden tabs don't poll; becoming visible ticks immediately to catch up;
//   - stop() tears the loop and the visibility listener down (in-flight work
//     is not cancelled — views keep their `stop` flag for detached-DOM writes).
//
// `kick()` runs a tick right now (e.g. after a user action that changed what
// the poll watches), subject to the same in-flight guard.
export function poller(fn, ms, { immediate = true } = {}) {
  let stopped = false;
  let inflight = false;

  async function tick(force) {
    if (stopped || inflight) return;
    if (!force && document.hidden) return;
    inflight = true;
    try {
      await fn();
    } catch {
      /* the view renders its own errors; keep polling */
    } finally {
      inflight = false;
    }
  }

  const timer = setInterval(tick, ms);
  const onVisibility = () => { if (!document.hidden) tick(); };
  document.addEventListener("visibilitychange", onVisibility);
  if (immediate) tick(true);

  return {
    kick: () => tick(true),
    stop() {
      stopped = true;
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    },
  };
}
