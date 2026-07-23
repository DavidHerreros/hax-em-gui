// Shared view helpers: error boxes, skeletons, and a tiny async loader.
import { h, mount } from "../h.js";
import { ApiError } from "../api.js";

export function errorBox(err) {
  let msg = String(err && err.message ? err.message : err);
  if (err instanceof ApiError && err.status === 401) {
    msg = "Unauthorized. Set your access token via the sidebar, then retry.";
  }
  return h("div", { class: "error-box" }, msg);
}

// Bytes as a human-readable size. Binary units, matching what a file manager shows
// (and what the server measures: apparent size, not blocks on disk).
export function formatSize(bytes) {
  if (bytes == null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let n = bytes, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

export function skeleton(height = "120px") {
  return h("div", { class: "skeleton", style: { height } });
}

export function skeletonGrid(n = 6) {
  return h("div", { class: "grid" }, ...Array.from({ length: n }, () => skeleton("130px")));
}

// Load async data into a container, showing a skeleton then the rendered result.
export function loadInto(container, loader, render, placeholder) {
  mount(container, placeholder || skeleton());
  loader()
    .then((data) => mount(container, render(data)))
    .catch((err) => mount(container, errorBox(err)));
  return container;
}
