// Update notifications for hax-em / hax-em-gui. The backend polls PyPI / git in
// the background and caches the result; here we just read that cache, keep a
// client-side copy with a tiny pub/sub, and render the home banner + the small
// badge that lives next to the sidebar brand.
import { h } from "./h.js";
import { icon } from "./icons.js";
import { api } from "./api.js";

const DISMISS_KEY = "hax_gui_dismissed_updates";
// How often the browser re-reads the server's cached status (cheap GET; the
// expensive PyPI/git work happens server-side on its own 6-hour schedule).
const CLIENT_REFRESH_MS = 10 * 60 * 1000;

let state = { checked_at: null, checking: false, packages: [] };
const subs = new Set();

export function subscribeUpdates(fn) {
  subs.add(fn);
  fn(state);
  return () => subs.delete(fn);
}

function emit() {
  for (const fn of subs) {
    try { fn(state); } catch { /* a subscriber failing must not break the rest */ }
  }
}

export async function refreshUpdates() {
  try {
    state = await api.updates();
    emit();
  } catch { /* offline / no auth — keep the last known state */ }
  return state;
}

// Force the server to re-probe PyPI/git *now* (not its 6-hour cache), then read
// the result. Used right after an update installs so the banner clears instead
// of lingering on the stale "update available" until the next poll.
export async function forceRefreshUpdates() {
  try {
    state = await api.checkUpdates();
    emit();
  } catch { return refreshUpdates(); }
  return state;
}

export function startUpdates() {
  // Warm-up: the server's first check (PyPI / git fetch) may not be done when
  // the page loads. Poll a few times until it has a result, then settle into
  // the slow refresh so the banner isn't stuck empty for the full interval.
  let tries = 0;
  const warmup = async () => {
    const s = await refreshUpdates();
    if (s.checked_at || ++tries >= 8) return;
    setTimeout(warmup, 3000);
  };
  warmup();
  setInterval(refreshUpdates, CLIENT_REFRESH_MS);
}

// -- dismissal (per package, keyed to the specific target so a *newer* update
// re-surfaces a previously dismissed banner) --------------------------------
function targetKey(p) {
  if (p.kind === "editable") return `${p.upstream || "upstream"}#${p.behind ?? 0}`;
  return p.latest || "";
}
function loadDismissed() {
  try { return JSON.parse(localStorage.getItem(DISMISS_KEY) || "{}"); } catch { return {}; }
}
function dismiss(p) {
  const d = loadDismissed();
  d[p.name] = targetKey(p);
  localStorage.setItem(DISMISS_KEY, JSON.stringify(d));
  emit();
}
function isDismissed(p) {
  return loadDismissed()[p.name] === targetKey(p);
}

function available(pkgs) {
  return (pkgs || []).filter((p) => p.update_available);
}

// -- describe one package's update in human terms ---------------------------
function describe(p) {
  if (p.kind === "editable") {
    const n = p.behind ?? 0;
    return `${p.branch || "your branch"} is ${n} commit${n === 1 ? "" : "s"} behind ${p.upstream || "upstream"}`;
  }
  if (p.kind === "vcs") return `a newer revision is available`;
  return `${p.current || "?"} → ${p.latest || "?"}`;
}

// ===== home banner =========================================================
export function UpdateBanner() {
  const el = h("div", { class: "update-banner-slot" });
  const render = (s) => {
    const items = available(s.packages).filter((p) => !isDismissed(p));
    el.replaceChildren();
    if (!items.length) return;
    el.appendChild(
      h("div", { class: "update-banner" },
        h("div", { class: "update-banner-icon" }, icon("rocket", 20)),
        h("div", { class: "update-banner-body" },
          h("h3", null, items.length === 1 ? "An update is available" : `${items.length} updates are available`),
          ...items.map(updateRow)),
      )
    );
  };
  subscribeUpdates(render);
  return el;
}

function updateRow(p) {
  const blockedDirty = p.kind === "editable" && p.dirty;
  return h("div", { class: "update-item" },
    h("div", { class: "update-item-head" },
      h("span", { class: "update-pkg" }, p.name),
      h("span", { class: "muted" }, describe(p)),
      h("button", { class: "update-dismiss", title: "Dismiss until the next update", onclick: () => dismiss(p) }, icon("x", 14)),
    ),
    blockedDirty
      ? h("p", { class: "update-warn" }, icon("ban", 13),
          " Working tree has uncommitted changes — commit or stash, then update with:")
      : null,
    blockedDirty ? h("pre", { class: "cmdline" }, p.command || "") : null,
    blockedDirty
      ? null
      : h("div", { class: "update-actions" },
          h("button", { class: "btn", onclick: () => openUpdate(p) }, icon("refresh", 15), "Update now"),
          h("span", { class: "update-cmd-hint muted" }, "or run: ",
            h("code", null, p.command || "")),
        ),
  );
}

function openUpdate(p) {
  // Lazy import breaks the updates.js <-> updateModal.js cycle (the modal calls
  // refreshUpdates from here when it closes).
  import("./views/updateModal.js").then((m) => m.updateModal(p));
}

// ===== sidebar brand badge =================================================
// Small pill showing how many updates are pending; links to Home where the
// banner explains each one. Hidden when nothing is pending.
export function updateBadge() {
  const el = h("a", { href: "#/", class: "update-badge", title: "Updates available" });
  const render = (s) => {
    const items = available(s.packages).filter((p) => !isDismissed(p));
    el.replaceChildren();
    if (!items.length) { el.style.display = "none"; return; }
    el.style.display = "inline-flex";
    el.appendChild(icon("rocket", 13));
    el.appendChild(h("span", null, String(items.length)));
  };
  subscribeUpdates(render);
  return el;
}
