// Update dialog. Two shapes share one streaming console:
//   * editable (devel) installs → a confirmation with three choices, because a
//     `git pull` can change pyproject.toml and a code pull alone won't install
//     new dependencies (and the JAX/CUDA stack is tightly pinned). The user
//     picks: don't update / pull and finish deps by hand / let the GUI do it all.
//   * pypi / vcs installs → a single "Update now" (pip resolves deps itself).
// For the GUI package an update ends by restarting (full) or closing (pull) the
// server; for hax-em it just updates in place.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { api, subscribeUpdateLog } from "../api.js";
import { createTerminal } from "./terminal.js";
import { refreshUpdates, forceRefreshUpdates } from "../updates.js";

export function updateModal(pkg) {
  const body = h("div", { class: "prompt-body" });
  const foot = h("div", { class: "modal-foot", style: { gap: "8px", flexWrap: "wrap" } });
  const modal = h("div", { class: "modal", style: { width: "min(720px, 94vw)" }, onclick: (e) => e.stopPropagation() },
    h("div", { class: "modal-head" },
      h("span", null, `Update ${pkg.name}`),
      h("button", { class: "btn-icon", style: { border: "none" }, onclick: () => close() }, icon("x", 18))),
    body, foot);
  const overlay = h("div", { class: "overlay", onclick: () => close() }, modal);
  document.body.appendChild(overlay);
  document.addEventListener("keydown", onKey);

  let term = null, closeLog = null, healthTimer = null, busy = false, locked = false;

  (pkg.kind === "editable" ? renderEditableConfirm : renderSimpleConfirm)();

  // ---- confirmation views ----
  function renderSimpleConfirm() {
    body.replaceChildren(
      h("p", { class: "muted", style: { marginTop: 0 } }, simpleSummary(pkg)),
      h("p", { class: "update-how" }, "Command:"),
      h("pre", { class: "cmdline" }, pkg.command || ""));
    foot.replaceChildren(
      h("button", { class: "btn secondary", onclick: () => close() }, "Cancel"),
      h("button", { class: "btn", onclick: () => start("full") }, icon("refresh", 16), "Update now"));
  }

  function renderEditableConfirm() {
    const n = pkg.behind ?? 0;
    const depChanged = !!pkg.pyproject_changed;
    body.replaceChildren(
      h("p", { class: "muted", style: { marginTop: 0 } },
        `Development install at `, h("code", null, pkg.repo || "your checkout"),
        ` on branch `, h("code", null, pkg.branch || "?"),
        ` — ${n} commit${n === 1 ? "" : "s"} behind ${pkg.upstream || "upstream"}.`),
      depChanged ? depWarning(pkg) : noDepNote(),
      pkg.is_gui ? serverNote() : null);

    const cancelBtn = h("button", { class: "btn secondary", onclick: () => close() }, "Not now");
    const pullBtn = h("button", { class: "btn secondary", onclick: () => start("pull") },
      icon("restore", 15), "Pull changes — I'll finish");
    const fullBtn = h("button", { class: "btn", onclick: () => start("full") },
      icon("refresh", 15), pkg.is_gui ? "Update everything & restart" : "Update everything");
    foot.replaceChildren(cancelBtn, h("div", { class: "ml-auto" }), pullBtn, fullBtn);
  }

  function depWarning(p) {
    return h("div", { class: "update-depbox" },
      h("p", null, icon("ban", 14), h("strong", null, " Dependencies may have changed."),
        " This update modifies ", h("code", null, "pyproject.toml"),
        ". A code pull alone won't install new dependencies, and the GPU stack is tightly pinned."),
      h("p", { class: "update-how" }, "Install them with:"),
      h("pre", { class: "cmdline" }, p.dep_command || ""),
      p.extra ? h("p", { class: "muted small" }, `Detected CUDA extra: `, h("code", null, `[${p.extra}]`), ".") : null);
  }
  function noDepNote() {
    return h("p", { class: "muted small" }, icon("check", 13),
      " No dependency changes detected in the incoming commits.");
  }
  function serverNote() {
    return h("p", { class: "muted small" },
      "Updating the GUI needs a server " +
      "restart (“update everything”) or close (“pull — I'll finish”, so you can install deps cleanly).");
  }

  // ---- running / streaming ----
  async function start(mode) {
    locked = true;
    const consoleEl = h("div", { class: "console", style: { marginTop: "4px" } });
    const note = h("p", { class: "update-modal-note muted" }, `Running …`);
    body.replaceChildren(note, consoleEl);
    foot.replaceChildren(h("span", { class: "muted small" }, "Working…"));
    try {
      await api.runUpdate(pkg.name, mode);
    } catch (err) {
      locked = false;
      note.textContent = err.detail || err.message || "could not start the update";
      note.className = "update-modal-note update-warn";
      foot.replaceChildren(h("button", { class: "btn secondary", onclick: () => close() }, "Close"));
      return;
    }
    busy = true;
    term = createTerminal(consoleEl);
    closeLog = subscribeUpdateLog((m) => onMessage(m, mode, note));
  }

  function onMessage(m, mode, note) {
    if (m.type === "snapshot") { term.reset(); if (m.data) term.write(m.data); }
    else if (m.type === "chunk") { term.write(m.data); }
    else if (m.type === "status" && m.status !== "running") finish(m, mode, note);
  }

  function finish(m, mode, note) {
    busy = false;
    locked = false;  // terminal state — re-locked below only while a restart is pending
    if (closeLog) { closeLog(); closeLog = null; }
    if (m.status === "failed") {
      note.textContent = m.note || "the update failed — see the log above";
      note.className = "update-modal-note update-warn";
      foot.replaceChildren(h("button", { class: "btn secondary", onclick: () => close() }, "Close"));
      return;
    }
    if (m.restarting) {                       // full update of the GUI package
      locked = true;                          // hold the modal until the reload
      note.textContent = "The server is restarting to load the new version. Reconnecting…";
      foot.replaceChildren(h("span", { class: "muted small" }, "Please wait…"));
      waitForServer(note);
      return;
    }
    if (m.closing) {                          // GUI pull: server steps aside
      showNextSteps(note, /*disconnected=*/true);
      return;
    }
    if (mode === "pull") {                     // hax-em pull: code in, deps are yours
      showNextSteps(note, /*disconnected=*/false);
      return;
    }
    // full update of hax-em (or GUI that couldn't self-restart)
    note.textContent = m.note || "Update installed.";
    note.className = "update-modal-note update-ok";
    forceRefreshUpdates();
    foot.replaceChildren(h("button", { class: "btn", onclick: () => close() }, icon("check", 16), "Done"));
  }

  // The manual finish: show the exact commands. For the GUI the page is about to
  // disconnect (server closing); for hax-em the GUI keeps running.
  function showNextSteps(note, disconnected) {
    note.textContent = "";
    note.className = "update-modal-note";
    const steps = [];
    if (pkg.pyproject_changed && pkg.dep_command) {
      steps.push(h("p", { class: "update-how" }, "1. Install the updated dependencies:"));
      steps.push(h("pre", { class: "cmdline" }, pkg.dep_command));
    }
    if (disconnected) {
      steps.push(h("p", { class: "update-how" }, `${pkg.pyproject_changed ? "2." : "1."} Start the server again:`));
      steps.push(h("pre", { class: "cmdline" }, pkg.start_command || "hax-gui start"));
    }
    body.appendChild(h("div", { class: "update-nextsteps" },
      h("p", { class: "update-modal-note update-ok" }, icon("check", 15),
        disconnected ? " Changes pulled. The server is closing — finish in your terminal:" : " Changes pulled. Finish in your terminal:"),
      ...(steps.length ? steps : [h("p", { class: "muted small" }, "No dependency changes — you're up to date.")])));
    if (disconnected) {
      locked = false;  // server is gone; allow dismissing
      foot.replaceChildren(h("span", { class: "muted small" }, "This page is now disconnected."));
    } else {
      forceRefreshUpdates();
      foot.replaceChildren(h("button", { class: "btn", onclick: () => close() }, icon("check", 16), "Done"));
    }
  }

  // ---- restart reconnect ----
  function waitForServer(note) {
    let tries = 0;
    const tick = async () => {
      tries += 1;
      try { await api.health(); term.write("\nServer is back — reloading.\n"); location.reload(); return; }
      catch { /* still down */ }
      if (tries > 60) {
        healthTimer = null; locked = false;
        note.textContent = "The server is taking a while to come back. Reload once it's up.";
        foot.replaceChildren(h("button", { class: "btn", onclick: () => location.reload() }, "Reload now"));
        return;
      }
      healthTimer = setTimeout(tick, 1500);
    };
    healthTimer = setTimeout(tick, 2500);
  }

  // ---- lifecycle ----
  function onKey(e) { if (e.key === "Escape" && !busy && !locked) close(); }
  function close() {
    if (busy || healthTimer || locked) return;  // don't bail mid-run / mid-restart
    if (closeLog) closeLog();
    overlay.remove();
    document.removeEventListener("keydown", onKey);
    refreshUpdates();
  }
}

function simpleSummary(pkg) {
  if (pkg.kind === "vcs") return "Reinstall the latest revision from its git source.";
  return `Upgrade ${pkg.name} from ${pkg.current || "?"} to ${pkg.latest || "?"} via pip.`;
}
