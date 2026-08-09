// Storage dashboard for one project: where the bytes are, run by run, plus what
// is reclaimable (failed/cancelled leftovers and the rebuildable SSD scratch).
// Data comes from /api/projects/{id}/usage — measured server-side off the event
// loop and cached, so reopening is instant and Refresh forces a re-walk.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { navigate } from "../router.js";
import { errorBox, formatSize } from "./ui.js";
import { badge } from "./jobs.js";

const RECLAIMABLE = new Set(["failed", "cancelled", "interrupted"]);

export function openStorageModal(project) {
  const body = h("div", { class: "storage-body" });
  const refreshBtn = h("button", { class: "btn-icon", title: "Re-measure (walks the project folder again)",
    onclick: () => load(true) }, icon("refresh", 15));
  const modal = h("div", { class: "modal storage-modal", onclick: (e) => e.stopPropagation() },
    h("div", { class: "modal-head" },
      h("span", null, "Storage — ", h("strong", null, project.name)),
      refreshBtn,
      h("button", { class: "btn-icon", style: { border: "none" }, onclick: close }, icon("x", 18))),
    body);
  const overlay = h("div", { class: "overlay", onclick: close }, modal);
  document.body.appendChild(overlay);
  document.addEventListener("keydown", onKey);
  load(false);

  function onKey(e) { if (e.key === "Escape") close(); }
  function close() { overlay.remove(); document.removeEventListener("keydown", onKey); }

  async function load(refresh) {
    refreshBtn.disabled = true;
    if (refresh) mount(body, h("p", { class: "muted", style: { padding: "16px" } }, "Measuring…"));
    else mount(body, h("p", { class: "muted", style: { padding: "16px" } }, "Reading sizes…"));
    try { render(await api.projectUsage(project.id, refresh)); }
    catch (err) { mount(body, errorBox(err)); }
    finally { refreshBtn.disabled = false; }
  }

  function render(u) {
    clear(body);
    const jobs = u.jobs || [];
    const maxBytes = Math.max(1, ...jobs.map((j) => j.bytes));
    const reclaimable = jobs.filter((j) => RECLAIMABLE.has(j.status))
                            .reduce((s, j) => s + j.bytes, 0);

    // headline numbers
    const stats = h("div", { class: "storage-stats" },
      stat("Project folder", formatSize(u.bytes), `${u.files} files`),
      u.scratch && u.scratch.exists ? stat("SSD scratch", formatSize(u.scratch.bytes),
        "rebuildable cache — reclaim first", true) : null,
      reclaimable ? stat("In failed/cancelled runs", formatSize(reclaimable),
        "delete those runs to reclaim", true) : null,
      u.other_bytes ? stat("Outside any run", formatSize(u.other_bytes),
        "project metadata + deleted-run leftovers") : null);
    body.appendChild(stats);

    if (!jobs.length) {
      body.appendChild(h("p", { class: "muted", style: { padding: "0 16px 16px" } }, "No runs with a folder yet."));
      return;
    }
    const list = h("div", { class: "storage-rows" });
    for (const j of jobs) list.appendChild(rowFor(j, maxBytes));
    body.appendChild(list);
    body.appendChild(h("p", { class: "muted small", style: { padding: "8px 16px 14px" } },
      "Sizes are the runs' folders. Deleting a run removes its folder, outputs and log — the job row disappears with it."));
  }

  function stat(label, value, hint, warn) {
    return h("div", { class: "storage-stat" + (warn ? " warn" : "") },
      h("div", { class: "muted small" }, label),
      h("div", { class: "mono storage-num" }, value),
      hint ? h("div", { class: "muted small" }, hint) : null);
  }

  function rowFor(j, maxBytes) {
    const active = j.status === "running" || j.status === "pending" || j.status === "cancelling" || j.status === "queued";
    const del = h("button", { class: "btn-icon", title: active ? "Running — cancel it before deleting" : "Delete this run and its folder",
      disabled: active,
      style: { color: active ? "" : "var(--err)" },
      onclick: async (e) => {
        e.stopPropagation();
        if (!window.confirm(`Delete run “${j.name || j.program}” and its folder (${formatSize(j.bytes)})? This cannot be undone.`)) return;
        del.disabled = true;
        try { await api.deleteJob(j.id); load(true); }
        catch (err) { del.disabled = false; alert(err && err.message ? err.message : "delete failed"); }
      } }, icon("trash", 14));
    return h("div", { class: "storage-row", onclick: () => { close(); navigate(`/jobs/${j.id}`); } },
      h("div", { class: "storage-row-main" },
        h("div", { class: "row", style: { gap: "8px", minWidth: "0" } },
          h("span", { class: "mono storage-name", title: j.workdir || "" }, j.name || j.program),
          j.name ? h("span", { class: "muted small mono" }, j.program) : null,
          badge(j.status)),
        h("div", { class: "storage-bar" },
          h("div", { class: "storage-bar-fill" + (RECLAIMABLE.has(j.status) ? " reclaim" : ""),
                     style: { width: `${Math.max(1, Math.round(100 * j.bytes / maxBytes))}%` } }))),
      h("span", { class: "mono storage-size" }, formatSize(j.bytes)),
      h("span", { class: "muted small storage-files" }, `${j.files} files`),
      del);
  }
}
