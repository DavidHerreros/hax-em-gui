// Live job view: streams the log over WebSocket into a terminal renderer
// (ANSI colors + progress lines), with cancel / rerun / delete actions.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api, subscribeLog } from "../api.js";
import { navigate, reload } from "../router.js";
import { store } from "../store.js";
import { errorBox, formatSize } from "./ui.js";
import { badge, staleBadge, jobDisplayName } from "./jobs.js";
import { openContextMenu } from "./contextMenu.js";
import { jobLabelChips } from "./labels.js";
import { launchViewerForJob } from "./viewerLaunch.js";
import { promptModal } from "./promptModal.js";
import { confirmJobCascade } from "./cascadeModal.js";
import { createTerminal } from "./terminal.js";
import { openFileBrowser } from "./filePicker.js";
import { loadProgramGroups, isPredictLatents } from "../programGroups.js";

export function JobDetailView(params) {
  const id = params.id;
  const root = h("div", { class: "view" });

  const head = h("div", { class: "row", style: { marginBottom: "8px" } });
  const progress = h("div", { class: "progress" });
  const cmd = h("pre", { class: "cmdline" }, "…");
  const wdLine = h("div");
  // How much this run has written. Measured on demand (it walks the run folder), not
  // polled: a long training keeps growing, so there is a refresh rather than a number
  // that silently ages. Re-measured for free when the run reaches a terminal status.
  const diskVal = h("span", { class: "mono" }, "…");
  const diskLine = h("div", { class: "row", style: { gap: "6px", marginTop: "2px" } });
  const wd = h("div", { class: "muted small", style: { margin: "-8px 0 16px" } }, wdLine, diskLine);
  const inputs = h("div", { class: "job-maps" });
  const maps = h("div", { class: "job-maps" });
  const actionErr = h("div", { style: { marginBottom: "12px" } });
  // Terminal output split into stdout / stderr tabs.
  const outConsole = h("div", { class: "console" });
  const errConsole = h("div", { class: "console", style: { display: "none" } });
  const errDot = h("span", { class: "tab-dot", title: "New stderr output" });
  const tabOut = h("div", { class: "tab active", onclick: () => selectStream("out") }, h("span", { class: "tab-label" }, "stdout"));
  const tabErr = h("div", { class: "tab", onclick: () => selectStream("err") }, h("span", { class: "tab-label" }, "stderr"), errDot);
  const consoleTabs = h("div", { class: "tabbar console-tabs" }, tabOut, tabErr);

  root.appendChild(h("a", { class: "back", href: "#/jobs" }, icon("arrowLeft", 16), "Jobs"));
  root.appendChild(head);
  root.appendChild(progress);
  root.appendChild(cmd);
  root.appendChild(wd);
  root.appendChild(inputs);
  root.appendChild(maps);
  root.appendChild(actionErr);
  root.appendChild(consoleTabs);
  root.appendChild(outConsole);
  root.appendChild(errConsole);

  const termOut = createTerminal(outConsole);
  const termErr = createTerminal(errConsole);
  let activeStream = "out";
  let status = "pending";
  let job = null;

  function selectStream(s) {
    activeStream = s;
    outConsole.style.display = s === "out" ? "" : "none";
    errConsole.style.display = s === "err" ? "" : "none";
    tabOut.classList.toggle("active", s === "out");
    tabErr.classList.toggle("active", s === "err");
    if (s === "err") tabErr.classList.remove("has-err");   // seen — clear the dot
  }
  // Flag the stderr tab when it gets output while it isn't the one on screen.
  function markErr(text) { if (text && activeStream !== "err") tabErr.classList.add("has-err"); }

  function renderHead() {
    clear(head);
    const isNamed = job && job.name;
    head.appendChild(h("h1", { style: { fontSize: "20px", color: "var(--accent)", fontFamily: "JetBrains Mono, monospace" } }, job ? jobDisplayName(job) : "…"));
    if (job) head.appendChild(h("button", { class: "row-menu", title: "Rename run", onclick: rename }, icon("pencil", 15)));
    if (isNamed) head.appendChild(h("span", { class: "muted mono", style: { fontSize: "12px" } }, job.program));
    head.appendChild(badge(status));
    if (job && job.stale) head.appendChild(staleBadge(job));
    if (job && job.project_name) head.appendChild(h("span", { class: "proj-chip" }, icon("box", 12), job.project_name));
    if (job) head.appendChild(jobLabelChips(job, { editable: true }));
    const running = status === "running" || status === "pending";
    // Being torn down: no longer cancellable, not finished either — the final
    // status arrives over the WS once the process group has been reaped.
    const cancelling = status === "cancelling";
    if (job && job.return_code != null && !running && !cancelling) {
      head.appendChild(h("span", { class: "muted", style: { fontSize: "12px" } }, "exit " + job.return_code));
    }
    if (job && status === "saved") {
      head.appendChild(h("div", { class: "row", style: { marginLeft: "auto", gap: "8px" } },
        h("button", { class: "btn", onclick: startSaved, title: "Launch this saved job now" }, icon("play", 15), "Run"),
        viewable(job) ? viewerBtn() : null,
        h("button", { class: "btn secondary", onclick: editRerun, title: "Open the form preloaded with these inputs to edit and save/launch a new run" }, icon("copy", 15), "Edit"),
        h("button", { class: "btn ghost", style: { color: "var(--err)" }, onclick: del }, icon("trash", 15), "Delete")));
    } else if (running && job) {
      head.appendChild(h("div", { class: "row", style: { marginLeft: "auto", gap: "8px" } },
        job.has_metrics ? metricsBtn() : null,
        viewable(job) ? viewerBtn() : null,
        h("button", { class: "btn ghost", style: { color: "var(--warn)" }, onclick: (e) => {
          // immediate feedback; the terminal status still arrives over the WS once
          // the process group has been torn down
          const b = e.currentTarget; b.disabled = true; b.lastChild.textContent = "Cancelling…";
          api.cancelJob(id).catch(() => { b.disabled = false; b.lastChild.textContent = "Cancel"; });
        } }, icon("ban", 15), "Cancel")));
    } else if (cancelling && job) {
      head.appendChild(h("div", { class: "row", style: { marginLeft: "auto", gap: "8px" } },
        job.has_metrics ? metricsBtn() : null,
        viewable(job) ? viewerBtn() : null,
        h("button", { class: "btn ghost", style: { color: "var(--warn)" }, disabled: true,
          title: "Stopping this run and its workers…" }, icon("ban", 15), "Cancelling…")));
    } else if (job) {
      // A cancelled/failed/interrupted run that already committed an intermediate
      // checkpoint can be resumed in place (backend sets job.resumable). Paired
      // actions (Continue/Rerun + their "Edit &" variant) collapse into split buttons.
      const canResume = job.resumable &&
        (status === "failed" || status === "cancelled" || status === "interrupted");
      head.appendChild(h("div", { class: "row", style: { marginLeft: "auto", gap: "8px" } },
        job.has_metrics ? metricsBtn() : null,
        viewable(job) ? viewerBtn() : null,
        canResume ? splitButton(
          { icon: "restore", label: "Continue", title: "Resume from the last saved checkpoint with the same inputs", onClick: continueRun },
          [{ icon: icon("sliders", 15), label: "Edit & continue", onClick: editContinue }]) : null,
        splitButton(
          { icon: "play", label: "Rerun", title: "Re-launch immediately with the same inputs (a fresh run)", onClick: rerun },
          [{ icon: icon("copy", 15), label: "Edit & rerun", onClick: editRerun }]),
        h("button", { class: "btn ghost", style: { color: "var(--err)" }, onclick: del }, icon("trash", 15), "Delete")));
    }
    renderProgress(running || cancelling);
  }

  function editRerun() {
    store.editFrom(job);
    navigate("/projects");
  }

  // Resume this run in place: continue_job relaunches the same job (same folder), so
  // the program picks up its checkpoint. Re-render the current route to reset the view.
  //
  // The folder survives, so downstream input *paths* stay valid — but what they point at
  // is about to be rewritten from a further-trained checkpoint, so anything derived from
  // this run is just as out of date as after a full rerun. Same dialog, different words.
  async function continueRun() {
    clear(actionErr);
    const choice = await confirmJobCascade(id, {
      title: "Continue this run?",
      verb: "Continue",
      lead: "It resumes from its last checkpoint and keeps writing into the same run folder, replacing its current outputs.",
    });
    if (!choice) return;
    try { await api.continueJob(id, choice); reload(); }
    catch (err) { mount(actionErr, errorBox(err)); }
  }

  // Same resume, but open the form first so the user can change parameters (e.g. a
  // smaller batch size to get past an OOM) before continuing from the checkpoint.
  function editContinue() {
    store.editFrom(job, { resume: true });
    navigate("/projects");
  }

  function viewable(j) {
    if (!j) return false;
    const a = j.args || {};
    // a finished FlexConsensus run is viewable: pick a consensus space (auto-decoder)
    if (j.program === "flexconsensus") return j.status === "succeeded";
    const reload = a.pickled_nn || a.reload;
    if (!reload) return false;
    if (a.z_space || a.latents || a.latents_file) return true;
    // predict runs expose latents as output once finished (which programs is
    // derived from the backend, see programGroups.js)
    return isPredictLatents(j.program) && j.status === "succeeded";
  }
  async function openViewer() {
    clear(actionErr);
    try { await launchViewerForJob(job); }   // opens in the floating viewer window
    catch (err) { mount(actionErr, errorBox(err)); }
  }

  // ── how much disk this run occupies ──
  function renderDiskLine() {
    clear(diskLine);
    if (!job || !job.workdir) return;
    diskLine.append(
      h("span", null, "Run folder: "),
      // shrinkable so a long path ellipsises instead of pushing the size off the row
      h("span", { class: "mono", title: job.workdir,
        style: { minWidth: "0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }, job.workdir),
      h("span", null, " · "),
      diskVal,
      h("button", { class: "btn-icon", title: "Re-measure this run's disk usage",
        onclick: () => loadUsage(true) }, icon("refresh", 12)));
  }
  async function loadUsage(refresh) {
    if (!job || !job.workdir) return;
    if (refresh) diskVal.textContent = "measuring…";
    try {
      const u = await api.jobUsage(id, refresh);
      diskVal.textContent = u.exists
        ? `${formatSize(u.bytes)} in ${u.files} file${u.files === 1 ? "" : "s"}`
        : "no run folder";
    } catch { diskVal.textContent = "size unavailable"; }
  }

  // ── outputs this run produced: inspect maps (.mrc) and metadata (.xmd/.star/.cs) ──
  async function loadOutputs() {
    const [m, t] = await Promise.allSettled([api.jobMrcs(id), api.jobMetadata(id)]);
    const mapsF = m.status === "fulfilled" ? (m.value.files || []) : [];
    const tablesF = t.status === "fulfilled" ? (t.value.files || []) : [];
    // While the run is active, keep the section (with a refresh) visible even when
    // empty, so files show up as they're written without reopening the view.
    const active = isActive();
    renderGroup(maps, "produced", mapsF, tablesF, loadOutputs, active);
  }
  async function loadInputs() {
    let mapsF = [], tablesF = [];
    try { const r = await api.jobInputs(id); mapsF = r.maps || []; tablesF = r.metadata || []; } catch { /* none */ }
    renderGroup(inputs, "inputs", mapsF, tablesF, loadInputs, false);
  }
  function renderGroup(container, kindWord, mapsF, tablesF, reload, showEmpty) {
    clear(container);
    const total = mapsF.length + tablesF.length;
    if (!total && !showEmpty) return;
    const parts = [];
    if (mapsF.length) parts.push(`${mapsF.length} map${mapsF.length > 1 ? "s" : ""}`);
    if (tablesF.length) parts.push(`${tablesF.length} metadata`);
    const label = total ? `${parts.join(" · ")} ${kindWord}` : `no ${kindWord} yet`;
    // The lists above only surface maps and metadata; the run folder holds checkpoints,
    // logs and everything else, so give it a way in. Outputs only — a run's inputs are
    // scattered wherever they came from, and there is no single folder to open.
    const browse = kindWord === "produced" && job && job.workdir
      ? h("button", { class: "btn-icon", title: "Browse this run's output folder — tick several maps or tables to open them together",
          onclick: () => openFileBrowser(job.workdir) }, icon("folderOpen", 14))
      : null;
    container.appendChild(h("div", { class: "row section-head", style: { margin: "0 0 8px" } },
      h("span", { class: "muted small" }, icon("box", 13), " ", label),
      h("span", { class: "ml-auto" }),
      browse,
      h("button", { class: "btn-icon", title: "Refresh", onclick: reload }, icon("refresh", 14))));
    if (total) {
      container.appendChild(h("div", { class: "map-list" },
        ...mapsF.map((f) => fileRow(f, "map")),
        ...tablesF.map((f) => fileRow(f, "table"))));
    } else {
      container.appendChild(h("div", { class: "muted small", style: { marginBottom: "10px" } }, "Maps and metadata appear here as the run writes them."));
    }
  }
  function fileRow(f, kind) {
    const isMap = kind === "map";
    const btn = h("button", { class: "btn small", title: isMap ? "Open this map in the 3D / slice viewer" : "Browse this metadata as a table (with plots)", onclick: () => inspectOutput(f, kind, btn) },
      icon(isMap ? "box" : "grid", 13), isMap ? "Inspect" : "Table");
    return h("div", { class: "map-row" },
      icon(isMap ? "box" : "grid", 13),
      h("span", { class: "mono", style: { flex: "1", minWidth: "0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, title: f.path }, f.name),
      h("span", { class: "muted small" }, formatSize(f.size)),
      btn);
  }
  async function inspectOutput(f, kind, btn) {
    clear(actionErr);
    const orig = btn ? btn.lastChild.textContent : "";
    if (btn) { btn.disabled = true; btn.lastChild.textContent = "Opening…"; }
    try {
      const v = kind === "map" ? await api.mrcViewerFromJob(id, f.path) : await api.mdViewerFromJob(id, f.path);
      store.openViewer(v);   // shows in the floating viewer window over this page
    } catch (err) { mount(actionErr, errorBox(err)); }
    finally { if (btn) { btn.disabled = false; btn.lastChild.textContent = orig; } }
  }
  const viewerBtn = () => h("button", { class: "btn secondary", onclick: openViewer, title: "Explore this run's conformational landscape in the embedded WebGL viewer" }, icon("box", 15), "3D viewer");

  // Reads the run's event files natively — so it works on a job that is still training.
  const metricsBtn = () => h("button", {
    class: "btn secondary", title: "Training curves, image panels and the latent space for this run",
    onclick: () => {
      store.openMetrics({ logdir: job.workdir, title: `${job.program} · ${jobDisplayName(job)}`, jobId: id });
      navigate("/metrics");
    },
  }, icon("chart", 15), "Metrics");

  // A primary action + a caret that opens its variant(s) as a menu, so paired actions
  // (e.g. Continue / Edit & continue) read as one control. `items` are context-menu
  // entries: { icon, label, onClick, danger? }.
  function splitButton(main, items) {
    return h("div", { class: "btn-split" },
      h("button", { class: "btn secondary split-main", title: main.title || "", onclick: main.onClick }, icon(main.icon, 15), main.label),
      h("button", { class: "btn secondary split-caret", title: "More…", onclick: (e) => { e.stopPropagation(); openContextMenu(e.currentTarget, items); } },
        h("span", { class: "split-chevron" }, icon("chevronUp", 13))));
  }

  async function startSaved() {
    clear(actionErr);
    try {
      const j = await api.startJob(id);
      job = j; status = j.status;
      termOut.reset(); termErr.reset();
      connectLog();      // re-open the stream now that the job is running
      renderHead();
      renderDiskLine();
    } catch (err) { mount(actionErr, errorBox(err)); }
  }

  async function rename() {
    if (!job) return;
    const name = await promptModal({ title: "Rename run", label: "Run name", value: job.name || "", placeholder: job.program, okText: "Rename" });
    if (name === null) return;
    try { const u = await api.renameJob(id, name); job = { ...job, name: u.name }; store.renameViewerForJob(id, u.name); renderHead(); }
    catch (err) { mount(actionErr, errorBox(err)); }
  }

  async function rerun() {
    clear(actionErr);
    const choice = await confirmJobCascade(id);
    if (!choice) return;
    // Rerun replaces this job in place (same id), so re-render the current route
    // to reset the log/state instead of navigating (the hash is unchanged).
    try { await api.rerunJob(id, choice); reload(); }
    catch (err) { mount(actionErr, errorBox(err)); }
  }

  async function del() {
    if (!window.confirm("Delete this job and its run folder (outputs + log)? This cannot be undone.")) return;
    clear(actionErr);
    try { await api.deleteJob(id); navigate("/jobs"); }
    catch (err) { mount(actionErr, errorBox(err)); }
  }

  function renderProgress(running) {
    clear(progress);
    if (status === "saved") return; // not started yet — nothing to show
    if (running) {
      progress.appendChild(h("div", { class: "sweep" }));
    } else {
      const kind = status === "succeeded" ? "succeeded" : status === "failed" ? "failed" : "other";
      progress.appendChild(h("div", { class: "bar " + kind, style: { width: "100%" } }));
    }
  }

  // Fetch job metadata (command + program name), then stream the log.
  api.job(id)
    .then((j) => {
      job = j; status = j.status;
      cmd.textContent = j.command.join(" ");
      if (j.cwd) wdLine.textContent = "Working directory: " + j.cwd;
      renderHead();
      renderDiskLine();
      loadUsage(false);
      loadInputs();
      loadOutputs();
      updatePolling();
      // groups drive the "viewable" check; re-render once they're known (no-op if
      // already cached, and the fallback keeps known programs correct meanwhile)
      loadProgramGroups().then(() => renderHead()).catch(() => {});
    })
    .catch((err) => mount(root, errorBox(err)));

  // Poll the run folder while the job is active so new outputs appear without
  // reopening the view; stop once it's finished.
  let poll = null;
  // 'cancelling' counts as active: the run is still being torn down and its final
  // status is still to come, so keep polling/streaming until it lands.
  const isActive = () => status === "running" || status === "pending" || status === "cancelling";
  function updatePolling() {
    const active = isActive();
    if (active && !poll) poll = setInterval(loadOutputs, 4000);
    else if (!active && poll) { clearInterval(poll); poll = null; }
  }

  let closeLog = null;
  function connectLog() {
    if (closeLog) closeLog();
    closeLog = subscribeLog(id, (m) => {
      if (m.type === "snapshot") {
        status = m.status;
        termOut.reset(); termErr.reset();
        const s = m.streams || {};
        if (s.out) termOut.write(s.out);
        if (s.err) { termErr.write(s.err); markErr(s.err); }
        renderHead(); loadOutputs(); updatePolling();
      }
      else if (m.type === "chunk") {
        if (m.stream === "err") { termErr.write(m.data); markErr(m.data); }
        else termOut.write(m.data);
      }
      else if (m.type === "status") {
        status = m.status;
        updatePolling();
        // A finished run's folder has stopped growing — worth one fresh measurement.
        if (m.status !== "running") api.job(id).then((j) => { job = j; renderHead(); loadOutputs(); if (!isActive()) loadUsage(true); }).catch(() => renderHead());
        else renderHead();
      }
    });
  }
  connectLog();

  return { el: root, cleanup: () => { if (closeLog) closeLog(); if (poll) clearInterval(poll); } };
}
