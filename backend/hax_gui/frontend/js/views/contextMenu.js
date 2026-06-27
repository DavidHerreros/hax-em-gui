// Tiny anchored popover menu (the "three-dots" menu). No dependencies: it mounts
// a full-screen transparent overlay that closes on outside-click / Escape, with
// the menu positioned just under the anchor element.
import { h } from "../h.js";

// items: [{ label, icon?, danger?, onClick }]
export function openContextMenu(anchorEl, items) {
  const menu = h("div", { class: "ctx-menu" },
    ...items.map((it) =>
      h("button", {
        class: "ctx-item" + (it.danger ? " danger" : ""),
        onclick: (e) => { e.stopPropagation(); close(); it.onClick(); },
      }, it.icon || null, h("span", null, it.label))));

  const overlay = h("div", {
    class: "ctx-overlay",
    onclick: close,
    oncontextmenu: (e) => { e.preventDefault(); close(); },
  }, menu);

  document.body.appendChild(overlay);
  document.addEventListener("keydown", onKey, true);

  // Position under the anchor, kept inside the viewport.
  const r = anchorEl.getBoundingClientRect();
  const mw = menu.offsetWidth || 180;
  const mh = menu.offsetHeight || 0;
  let left = r.right - mw;                 // right-align the menu to the button
  left = Math.max(8, Math.min(left, window.innerWidth - mw - 8));
  let top = r.bottom + 4;
  if (top + mh > window.innerHeight - 8) top = Math.max(8, r.top - mh - 4); // flip up
  menu.style.left = left + "px";
  menu.style.top = top + "px";

  function close() { overlay.remove(); document.removeEventListener("keydown", onKey, true); }
  function onKey(e) { if (e.key === "Escape") { e.stopPropagation(); close(); } }
}
