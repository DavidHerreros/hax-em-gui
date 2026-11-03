// Tiny hash-based router. Hash routing means the server needs no SPA fallback:
// every client path is just "/" to the backend. A view is a function
// (params) -> Node | { el: Node, cleanup?: () => void }.
import { clear } from "./h.js";
import { confirmModal } from "./views/confirmModal.js";

const routes = [];
let outlet = null;
let onNavigate = null;
let currentCleanup = null;
let navGuard = null;      // () => string | null — a message blocks navigation
let lastPath = null;
// True while the guard's confirm modal is open. A second hashchange landing in
// that window (another click, back/forward) must not stack a second modal or
// navigate out from under the first — it just reverts and waits.
let guardAsking = false;

// Views with unsaved state register a guard; navigating away asks first.
// Cleared automatically on every successful navigation (a stale guard from a
// torn-down view must never block the next one).
export function setNavGuard(fn) { navGuard = fn; }

export function defineRoutes(list) { routes.push(...list); }
export function setOutlet(el) { outlet = el; }
export function onRoute(cb) { onNavigate = cb; }
export function navigate(path) { location.hash = path; }
export function currentPath() { return location.hash.replace(/^#/, "") || "/"; }
// Re-render the current route (e.g. after an in-place change where the hash
// doesn't change, so "hashchange" wouldn't fire).
export function reload() { resolve(); }

function compile(pattern) {
  const names = [];
  const re = new RegExp(
    "^" + pattern.replace(/:[^/]+/g, (m) => { names.push(m.slice(1)); return "([^/]+)"; }) + "$"
  );
  return { re, names };
}

function render(route, params, path) {
  if (currentCleanup) { try { currentCleanup(); } catch { /* ignore */ } currentCleanup = null; }
  const result = route.view(params) || {};
  const node = result instanceof Node ? result : result.el;
  currentCleanup = result instanceof Node ? null : result.cleanup || null;
  clear(outlet);
  if (node) outlet.appendChild(node);
  if (onNavigate) onNavigate(path);
}

// The route-matching tail, factored out so both the guardless path and the
// confirmed-async path (below) can reach it without duplicating the loop.
function matchAndRender(path) {
  navGuard = null;
  lastPath = path;
  for (const r of routes) {
    const c = r._c || (r._c = compile(r.path));
    const m = path.match(c.re);
    if (m) {
      const params = {};
      c.names.forEach((n, i) => (params[n] = decodeURIComponent(m[i + 1])));
      return render(r, params, path);
    }
  }
  render(routes[0], {}, path);
}

function resolve() {
  const path = currentPath();
  if (guardAsking) {
    // A modal is already up for an earlier navigation attempt; this one just
    // reverts the hash back to where we're staying and waits for the answer.
    history.replaceState(null, "", "#" + lastPath);
    return;
  }
  if (navGuard && lastPath !== null && path !== lastPath) {
    const msg = navGuard();
    if (msg) {
      // The hash has already changed by the time resolve() runs, so revert it
      // first (staying put while asking) — then ask asynchronously; the
      // hashchange this replaceState would otherwise fire is suppressed.
      history.replaceState(null, "", "#" + lastPath);
      guardAsking = true;
      confirmModal({ title: "Unsaved changes", message: msg, okText: "Leave", cancelText: "Stay", danger: true })
        .then((ok) => {
          guardAsking = false;
          if (ok) {
            history.replaceState(null, "", "#" + path);
            matchAndRender(path);
          }
          // Cancelled: already reverted to lastPath above, nothing else to do.
        });
      return;
    }
  }
  matchAndRender(path);
}

export function startRouter() {
  window.addEventListener("hashchange", resolve);
  resolve();
}
