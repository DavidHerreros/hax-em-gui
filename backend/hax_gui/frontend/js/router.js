// Tiny hash-based router. Hash routing means the server needs no SPA fallback:
// every client path is just "/" to the backend. A view is a function
// (params) -> Node | { el: Node, cleanup?: () => void }.
import { clear } from "./h.js";

const routes = [];
let outlet = null;
let onNavigate = null;
let currentCleanup = null;

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

function resolve() {
  const path = currentPath();
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

export function startRouter() {
  window.addEventListener("hashchange", resolve);
  resolve();
}
