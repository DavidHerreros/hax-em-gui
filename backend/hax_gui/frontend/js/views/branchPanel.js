// Developer panel: git branch status & switching for hax source checkouts.
// Read-mostly — shows current branch, ahead/behind and a clean/dirty flag, lists
// the available branches, and switches between them only when the tree is clean.
// Every watched package is shown: one with a resolved checkout (editable, local,
// or a configured clone) gets a full card; one without gets a "set source
// folder" prompt so you can point it at a dev clone (e.g. a PyPI-installed
// hax-em you also develop in a git checkout).
import { h } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errorBox } from "./ui.js";
import { openFilePicker } from "./filePicker.js";

export function BranchPanel() {
  const root = h("div");
  load();
  return root;

  function load() {
    api.branches()
      .then(render)
      .catch(() => root.replaceChildren());  // no auth / error → just hide the panel
  }

  function render(pkgs) {
    if (!pkgs || !pkgs.length) { root.replaceChildren(); return; }
    root.replaceChildren(
      h("h2", { class: "section", style: { marginTop: "28px" } }, "Developer — git branches"),
      h("p", { class: "field-help", style: { margin: "0 0 12px" } },
        "Manage the branch of your hax source checkouts. An ",
        h("code", null, "pip install -e"), " install is found automatically; for a PyPI/conda install you can point at a separate dev clone."),
      ...pkgs.map((p) => (p.available ? branchCard(p) : sourceStub(p))));
  }

  // ---- package with no resolved checkout: offer to set one ----
  function sourceStub(p) {
    const msg = h("p", { class: "field-help", style: { margin: "0 0 12px" } },
      `No source checkout found for ${p.name} (it isn't a development install). Point at a git clone to manage its branch.`);
    const btn = h("button", { class: "btn secondary", onclick: pick }, icon("folderOpen", 15), "Set source folder");
    function pick() { openFilePicker("directory", (path) => save(p, path, msg)); }
    return h("div", { class: "card branch-card" },
      h("div", { class: "branch-head" }, h("span", { class: "update-pkg" }, p.name)),
      msg, btn);
  }

  async function save(p, path, msgEl) {
    if (!path) return;
    try { await api.setBranchSource(p.name, path); load(); }
    catch (err) { msgEl.replaceWith(h("p", { class: "update-warn small" }, err.detail || err.message || "could not set source")); }
  }

  // ---- package with a checkout: full branch card ----
  function branchCard(p) {
    const out = h("div", { class: "branch-out" });
    const select = h("select", { class: "gpu-select" },
      ...p.branches.map((b) => h("option", { value: b.name, selected: b.current || undefined },
        b.current ? `${b.name} (current)` : b.name + (b.where === "remote" ? "  ·  remote" : ""))));

    const switchBtn = h("button", { class: "btn", onclick: doSwitch }, icon("link", 15), "Switch");
    const refreshBtn = h("button", { class: "btn secondary", title: "Fetch remote branches", onclick: doRefresh },
      icon("refresh", 15), "Refresh");
    const syncSwitch = () => { switchBtn.disabled = p.dirty || select.value === p.branch; };
    select.addEventListener("change", syncSwitch);

    async function doSwitch() {
      const branch = select.value;
      switchBtn.disabled = refreshBtn.disabled = true;
      out.replaceChildren(h("p", { class: "muted small" }, `Switching to ${branch}…`));
      try {
        const r = await api.switchBranch(p.name, branch);
        out.replaceChildren(h("pre", { class: "cmdline" }, (r.output || "").trim() || `Switched to ${r.branch}.`));
        if (r.live && p.is_gui) afterGuiSwitch(out);
        else if (!r.live) showReinstall(out, r.reinstall_command);
        else load();  // live hax-em: takes effect on the next run
      } catch (err) {
        refreshBtn.disabled = false;
        out.replaceChildren(h("p", { class: "update-warn small" }, err.detail || err.message || "switch failed"));
        syncSwitch();
      }
    }

    async function doRefresh() {
      refreshBtn.disabled = true;
      out.replaceChildren(h("p", { class: "muted small" }, "Fetching remote branches…"));
      try { await api.fetchBranches(p.name); load(); }
      catch (err) { refreshBtn.disabled = false; out.replaceChildren(errorBox(err)); }
    }

    syncSwitch();
    return h("div", { class: "card branch-card" },
      h("div", { class: "branch-head" },
        h("span", { class: "update-pkg" }, p.name),
        h("code", { class: "branch-repo muted" }, p.repo)),
      h("p", { class: "branch-source muted small" }, sourceLabel(p),
        p.configured ? configuredControls(p) : null),
      h("div", { class: "branch-status" },
        h("span", { class: "branch-now" }, icon("workflow", 14), p.branch || "?"),
        aheadBehind(p),
        p.dirty
          ? h("span", { class: "branch-flag dirty" }, icon("ban", 13), " uncommitted changes")
          : h("span", { class: "branch-flag clean" }, icon("check", 13), " clean")),
      p.dirty ? h("p", { class: "field-help", style: { margin: "8px 0 0" } },
        "Commit or stash your changes to switch branches.") : null,
      h("div", { class: "branch-controls" }, select, switchBtn, refreshBtn),
      out);
  }

  function configuredControls(p) {
    const change = h("a", { class: "branch-link", onclick: () => openFilePicker("directory", (path) => save(p, path, null)) }, "change");
    const clear = h("a", { class: "branch-link", onclick: async () => { await api.setBranchSource(p.name, null); load(); } }, "clear");
    return h("span", null, "  (", change, " · ", clear, ")");
  }

  function sourceLabel(p) {
    if (p.source === "editable") return "Editable install — switches apply live.";
    if (p.source === "local-install") return "Local install — reinstall after switching to apply.";
    if (p.source === "configured") return "Configured clone — installed package is from elsewhere; reinstall to apply.";
    return "";
  }

  function showReinstall(out, cmd) {
    if (!cmd) { forceReload(); return; }
    out.appendChild(h("div", { style: { marginTop: "8px" } },
      h("p", { class: "update-how" }, "Reinstall to apply the switch:"),
      h("pre", { class: "cmdline" }, cmd),
      h("p", { class: "muted small" }, "Run it in your hax environment, then restart any affected runs.")));
  }
  function forceReload() { load(); }

  function aheadBehind(p) {
    if (p.upstream == null) return h("span", { class: "muted small" }, "no upstream");
    const bits = [];
    if (p.ahead) bits.push(`${p.ahead} ahead`);
    if (p.behind) bits.push(`${p.behind} behind`);
    return h("span", { class: "muted small" }, bits.length ? bits.join(", ") + ` of ${p.upstream}` : `up to date with ${p.upstream}`);
  }

  // Switching the GUI's own branch changes code the running server already
  // loaded — offer an in-place restart (deps from the new branch may still need
  // reinstalling; the update banner will flag that after the reload).
  function afterGuiSwitch(out) {
    const note = h("p", { class: "muted small", style: { margin: "8px 0 0" } });
    const btn = h("button", { class: "btn", style: { marginTop: "8px" }, onclick: restart }, icon("refresh", 15), "Restart server");
    out.appendChild(h("div", null,
      h("p", { class: "update-modal-note update-ok", style: { margin: "6px 0 0" } },
        icon("check", 14), " Branch switched. Restart the server to load it."),
      btn, note));

    async function restart() {
      btn.disabled = true;
      try {
        const r = await api.restartServer();
        if (!r.scheduled) {
          note.textContent = "Can't self-restart (not started via `hax-gui start`). Restart manually: hax-gui restart";
          return;
        }
      } catch { /* the server may already be going down */ }
      note.textContent = "Restarting — the page will reload when it's back…";
      let tries = 0;
      const tick = async () => {
        tries += 1;
        try { await api.health(); location.reload(); return; } catch { /* down */ }
        if (tries > 60) { note.textContent = "Taking a while — reload manually once it's up."; return; }
        setTimeout(tick, 1500);
      };
      setTimeout(tick, 2500);
    }
  }
}
