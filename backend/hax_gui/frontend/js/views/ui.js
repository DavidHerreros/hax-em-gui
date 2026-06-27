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
