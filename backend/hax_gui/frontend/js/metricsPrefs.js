// How a run's metrics were left: which sub-tab was open, which tags are plotted, the
// display controls, each card's y-scale / expand / solo / curve colours, and the image
// tab's sheet settings.
//
// A metrics panel is destroyed whenever you navigate off /metrics (the router calls the
// view's cleanup) and rebuilt from scratch on the way back, so without this every card
// forgets it was in Split, every curve forgets it was soloed, and the tag you unticked
// comes back. Kept in localStorage rather than in the store, so the configuration also
// survives a reload of the GUI — it describes a run you are reading, not a session.
//
// Keyed by log directory. Runs are evicted oldest-touched-first: a lab machine that has
// opened hundreds of runs should not carry all of them in localStorage forever.

const KEY = "hax_gui_metrics_prefs";
const MAX_RUNS = 32;

let all = read();

function read() {
  try {
    const parsed = JSON.parse(localStorage.getItem(KEY) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch { return {}; }
}

function write() {
  try { localStorage.setItem(KEY, JSON.stringify(all)); }
  catch { /* quota or private mode: the panel still works, it just forgets */ }
}

function trim() {
  const keys = Object.keys(all);
  if (keys.length <= MAX_RUNS) return;
  keys.sort((a, b) => (all[a].t || 0) - (all[b].t || 0))
    .slice(0, keys.length - MAX_RUNS)
    .forEach((k) => delete all[k]);
}

// One run's slice of the store. `section` is "panel" | "scalars" | "images" | "latent",
// so one view rewriting its own settings can never clobber another's.
export function prefsFor(logdir) {
  const entry = all[logdir] || (all[logdir] = { t: Date.now() });
  return {
    get(section) {
      const v = entry[section];
      return v && typeof v === "object" ? v : {};
    },
    set(section, value) {
      entry[section] = value;
      entry.t = Date.now();
      // An eviction while this panel was open would have detached our entry; putting it
      // back is what keeps a long-lived tab writing somewhere that is actually saved.
      if (all[logdir] !== entry) all[logdir] = entry;
      trim();
      write();
    },
    clear() {
      delete all[logdir];
      write();
    },
  };
}

// Only write when something actually changed: every poll re-runs the render path, and
// re-serialising identical settings into localStorage five times a minute is pure churn.
export function changeWriter(prefs, section) {
  let last = null;
  let timer = null;
  return function schedule(build, delay = 400) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      const value = build();
      const json = JSON.stringify(value);
      if (json === last) return;
      last = json;
      prefs.set(section, value);
    }, delay);
  };
}
