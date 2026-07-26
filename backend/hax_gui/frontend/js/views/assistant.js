// AI assistant chat panel: a chat-history sidebar (new / rename / delete, grouped
// by project) beside a streaming conversation with a live activity trace, Markdown
// rendering, and a Stop button. Only routed when /api/assistant/status reports the
// optional hax-em-assistant package is installed (see app.js).
//
// The conversation + WebSocket + streaming buffers live in a module-level
// controller (`ctrl`) that OUTLIVES the route view, so a question keeps running in
// the background when you leave the page: on return the view rebuilds from `ctrl`
// and you see the ongoing answering process — or the finished answer.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { assistantChat, api } from "../api.js";
import { renderMarkdown } from "../markdown.js";
import { store } from "../store.js";
import { createTerminal } from "./terminal.js";
import { runBrowser, folderName } from "./runBrowser.js";
import { jobLabelChips } from "./labels.js";
import { jobTimeEl } from "./jobs.js";

const GREETING =
  "Hi! I can help you pick programs and parameters, explain options from the docs, "
  + "diagnose a failed run's log, and reason about results. Ask me anything about your hax workflows.";

// ── persistent controller (survives view unmount) ───────────────────────────
const ctrl = {
  history: [],             // {role, content} — the active conversation
  currentChatId: null,     // persisted chat id (null = unsaved new chat)
  currentChatProject: "",  // project_id of the active chat (scopes the run picker)
  newChatProject: "",      // project_id for the next new chat
  attachedRun: null,       // {job_id,label,has_*} attached-run context
  grounded: true,
  // live turn state (an in-flight answer)
  streaming: false,
  liveRaw: "",             // partial assistant answer accumulated so far
  liveSteps: [],           // activity-trace step texts for the current turn
  liveActions: [],         // proposed-action records for this turn
  stopped: false,
  socket: null,            // one WebSocket for the app's lifetime (reused per turn)
  hooks: null,             // set by the mounted view; null while unmounted
};

function ensureSocket() {
  if (ctrl.socket) return ctrl.socket;
  ctrl.socket = assistantChat({
    onStatus: (text) => {
      if (!ctrl.streaming) return;
      ctrl.liveSteps.push(text);
      if (ctrl.hooks) ctrl.hooks.onStep(text);
    },
    onAction: (action) => {
      const rec = { action };
      ctrl.liveActions.push(rec);
      if (ctrl.hooks) ctrl.hooks.onAction(rec);
    },
    onToken: (t) => {
      if (!ctrl.streaming) return;
      ctrl.liveRaw += t;
      if (ctrl.hooks) ctrl.hooks.onToken();
    },
    onDone: () => { if (ctrl.streaming) completeTurn(); },
    onError: (err) => {
      if (!ctrl.streaming) return;
      ctrl.liveRaw += (ctrl.liveRaw ? "\n\n" : "") + `⚠ ${err}`;
      completeTurn();
    },
  });
  return ctrl.socket;
}

function sendMessage(text) {
  ctrl.history.push({ role: "user", content: text });
  ctrl.streaming = true;
  ctrl.liveRaw = "";
  ctrl.liveSteps = [];
  ctrl.liveActions = [];
  ctrl.stopped = false;
  const payload = { messages: ctrl.history, grounded: ctrl.grounded };
  if (ctrl.attachedRun) payload.run_job_id = ctrl.attachedRun.job_id;
  ensureSocket().send(payload);
  if (ctrl.hooks) ctrl.hooks.onSend();
}

function completeTurn() {
  if (ctrl.liveRaw) ctrl.history.push({ role: "assistant", content: ctrl.liveRaw });
  else if (!ctrl.stopped && ctrl.history.length &&
           ctrl.history[ctrl.history.length - 1].role === "user") ctrl.history.pop();
  ctrl.streaming = false;
  ctrl.liveRaw = "";
  if (ctrl.hooks) ctrl.hooks.onDone();
  persistCurrent();
}

function stopTurn() {
  if (!ctrl.streaming) return;
  ctrl.stopped = true;
  if (ctrl.socket) ctrl.socket.stop();
  ctrl.liveSteps.push("⏹ stopped");
  if (ctrl.hooks) ctrl.hooks.onStep("⏹ stopped");
  completeTurn();
}

function deriveTitle() {
  const u = ctrl.history.find((m) => m.role === "user");
  return u ? u.content.slice(0, 60) : "New chat";
}
async function persistCurrent() {
  if (!ctrl.history.length) return;
  try {
    if (!ctrl.currentChatId) {
      const c = await api.createChat(ctrl.newChatProject || null, deriveTitle());
      ctrl.currentChatId = c.id;
    }
    await api.updateChat(ctrl.currentChatId, { messages: ctrl.history });
    if (ctrl.hooks && ctrl.hooks.onChatsChanged) ctrl.hooks.onChatsChanged();
  } catch { /* persistence is best-effort */ }
}

// ── the route view (rebuilds from ctrl on mount; detaches on unmount) ────────
export function AssistantView() {
  let chats = [];              // chat metadata for the sidebar
  let projects = [];           // {id, name} for grouping + the new-chat selector
  let jobsCache = [];          // jobs for the attach-run picker
  let reportBusy = false;      // a PDF report is being generated (view-local UI)
  let dragChatId = null;       // chat being dragged onto a project group (move)

  // live-turn DOM refs (recreated on mount / on send)
  let current = null, currentActivity = null, raf = 0;

  // ── elements ──────────────────────────────────────────────────────────
  const log = h("div", { class: "chat-log" });
  const input = h("textarea", {
    class: "chat-input",
    rows: "1",
    placeholder: "Ask about programs, parameters, a run, a log… (Enter to send, Shift+Enter for newline)",
  });
  const sendBtn = h("button", { class: "btn chat-send" }, "Send");
  sendBtn.addEventListener("click", () => { if (ctrl.streaming) stopTurn(); else send(); });
  function setComposerMode() {
    sendBtn.textContent = ctrl.streaming ? "Stop" : "Send";
    sendBtn.classList.toggle("danger", ctrl.streaming);
  }

  const groundBtn = h("button", {
    class: "chip",
    title: "Ground answers in the hax docs + program schema",
    onclick: () => { ctrl.grounded = !ctrl.grounded; groundBtn.classList.toggle("active", ctrl.grounded); },
  }, "docs grounding");
  groundBtn.classList.toggle("active", ctrl.grounded);

  const statusText = h("span", null, "");
  // Rebuild the docs index in place so edits to hax-em-docs are picked up
  // without restarting the GUI. The index also auto-rebuilds on process start
  // whenever the docs content hash changes; this is the live path.
  const reindexBtn = h("button", {
    class: "chat-reindex btn-icon", title: "Reindex docs (pick up hax-em-docs edits without a restart)",
    onclick: async () => {
      reindexBtn.disabled = true;
      const prev = statusText.textContent;
      statusText.textContent = "reindexing docs…";
      try {
        const r = await api.assistantReindexDocs();
        if (r.ok) {
          statusText.textContent = `docs reindexed · ${r.chunks} doc chunks`;
          api.assistantStatus().then(showStatus).catch(() => {});
        } else {
          statusText.textContent = `reindex failed: ${r.reason || "unknown"}`;
        }
      } catch (e) {
        statusText.textContent = prev;
      } finally {
        reindexBtn.disabled = false;
      }
    },
  }, icon("refresh", 13));
  const statusLine = h("div", { class: "chat-status" }, statusText, reindexBtn);
  function showStatus(s) {
    statusText.textContent = s.available
      ? `model: ${s.model}  ·  docs: ${s.docs_grounding ? "on" : "off"}  ·  ${s.programs} programs, ${s.doc_chunks} doc chunks`
      : "assistant unavailable";
    reindexBtn.style.display = s.available ? "" : "none";
  }
  api.assistantStatus().then(showStatus).catch(() => { statusText.textContent = ""; });

  // ── attach a run (explicit): scope to the chat's project when it has one ──
  const attachRow = h("div", { class: "chat-attach" });
  async function loadJobs() {
    try { jobsCache = await api.jobs(); } catch { jobsCache = []; }
  }
  // ── run picker (modal): every run grouped by project, with filters + labels,
  //    so a run is easy to find by project / program / name / folder. ──────────
  let runPicker = null;
  function closeRunPicker() {
    if (!runPicker) return;
    runPicker.remove();
    document.removeEventListener("keydown", onRunPickerKey);
    runPicker = null;
  }
  function onRunPickerKey(e) { if (e.key === "Escape") closeRunPicker(); }
  function attachRunRow(j) {
    return h("button", { class: "run-row", onclick: () => { closeRunPicker(); attachJob(j.id); } },
      h("span", { class: "run-prog mono" }, j.name || j.program),
      j.name ? h("span", { class: "run-sub mono" }, j.program) : null,
      h("span", { class: "run-folder mono" }, folderName(j.workdir)),
      (j.status && j.status !== "succeeded") ? h("span", { class: "run-status" }, j.status) : null,
      (j.labels && j.labels.length) ? jobLabelChips(j) : null,
      jobTimeEl(j, "run-time"),
      h("span", { class: "run-open" }, icon("paperclip", 14), "Attach"));
  }
  function openRunPicker() {
    const browser = runBrowser({
      renderRow: attachRunRow,
      emptyText: "No runs yet. Launch a program, then attach its run here.",
    });
    const modal = h("div", { class: "modal run-picker", onclick: (e) => e.stopPropagation() },
      h("div", { class: "modal-head" }, h("span", null, "Attach a run"),
        h("button", { class: "btn-icon", style: { border: "none" }, onclick: closeRunPicker }, icon("x", 18))),
      h("div", { class: "run-picker-list" }, browser.el));
    runPicker = h("div", { class: "overlay", onclick: closeRunPicker }, modal);
    document.body.appendChild(runPicker);
    document.addEventListener("keydown", onRunPickerKey);
    browser.setStatus(h("p", { class: "muted", style: { padding: "12px" } }, "Loading runs…"));
    // Refresh on open so a just-finished run shows; the cache is the fallback.
    loadJobs().then(() => { if (runPicker) browser.setJobs(jobsCache); });
  }
  function renderAttachRow() {
    attachRow.replaceChildren();
    if (ctrl.attachedRun) {
      const ar = ctrl.attachedRun;
      const tags = ["metrics", "latents", "volumes"].filter((k) => ar["has_" + k])
        .map((k) => (k === "latents" && ar.latents_from) ? `latents ← ${ar.latents_from}` : k);
      const hasArtifacts = ["metrics", "latents", "volumes"].some((k) => ar["has_" + k]);
      const reportBtn = reportBusy
        ? h("span", { class: "chat-attach-busy" }, "Generating PDF…")
        : h("button", {
            class: "chat-item-btn", title: "Generate a PDF report of this run",
            disabled: hasArtifacts ? undefined : "true",
            onclick: downloadReport,
          }, icon("fileText", 14));
      const chip = h("div", { class: "chat-attach-chip" },
        icon("paperclip", 14),
        h("span", { class: "chat-attach-name" }, ar.label),
        h("span", { class: "chat-attach-tags" },
          tags.length ? tags.join(" · ") : "no readable artifacts"),
        reportBtn,
        h("button", { class: "chat-item-btn", title: "Detach run", onclick: detachRun }, icon("x", 14)));
      attachRow.appendChild(chip);
      return;
    }
    attachRow.appendChild(h("span", { class: "muted chat-attach-lead" }, icon("paperclip", 13), "Run"));
    attachRow.appendChild(h("button", {
      class: "chat-attach-btn",
      title: "Attach a run so the assistant can summarize its metrics, landscape and volume motions",
      onclick: openRunPicker,
    }, icon("paperclip", 13), "Attach a run…"));
  }
  async function attachJob(id) {
    if (!id) { ctrl.attachedRun = null; renderAttachRow(); return; }
    try { ctrl.attachedRun = await api.assistantRunPreview(id); }
    catch { ctrl.attachedRun = { job_id: id, label: "run", has_metrics: false, has_latents: false, has_volumes: false }; }
    renderAttachRow();
  }
  function detachRun() { ctrl.attachedRun = null; renderAttachRow(); }
  async function downloadReport() {
    if (!ctrl.attachedRun || reportBusy) return;
    reportBusy = true; renderAttachRow();
    try {
      await api.assistantRunReport(ctrl.attachedRun.job_id, true);
    } catch (e) {
      window.alert("Report generation failed: " + (e && (e.detail || e.message) || "unknown error"));
    } finally {
      reportBusy = false; renderAttachRow();
    }
  }

  const chatList = h("div", { class: "chat-list" });
  const projectSelect = h("select", {
    class: "chat-project-select",
    onchange: (e) => {
      ctrl.newChatProject = e.target.value;
      // an unsaved chat follows the picker's project scope immediately
      if (!ctrl.currentChatId) { ctrl.currentChatProject = ctrl.newChatProject; renderAttachRow(); }
    },
  });

  // ── conversation rendering ────────────────────────────────────────────
  function bubble(role, text) {
    const body = h("div", { class: "chat-body" });
    if (role === "user") body.textContent = text;
    else body.replaceChildren(renderMarkdown(text));
    log.appendChild(h("div", { class: `chat-msg ${role}` },
      h("div", { class: "chat-role" }, role === "user" ? "You" : "Assistant"), body));
    log.scrollTop = log.scrollHeight;
    return body;
  }
  // Rebuild the whole log from ctrl: history, the in-flight turn (if streaming),
  // and any proposed-action cards. Called on mount and on chat switch.
  function renderConversation() {
    log.replaceChildren();
    current = currentActivity = null;
    if (!ctrl.history.length && !ctrl.streaming) { bubble("assistant", GREETING); }
    else for (const m of ctrl.history) bubble(m.role, m.content);
    if (ctrl.streaming) {
      startLiveTurn();
      for (const s of ctrl.liveSteps) appendStep(s);
      if (ctrl.liveRaw) current.replaceChildren(renderMarkdown(ctrl.liveRaw));
    }
    for (const rec of ctrl.liveActions) renderActionCard(rec.action);
    log.scrollTop = log.scrollHeight;
    setComposerMode();
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });

  // ── live turn DOM ─────────────────────────────────────────────────────
  function startLiveTurn() {
    const activity = h("div", { class: "chat-activity" });
    const body = h("div", { class: "chat-body streaming" });
    log.appendChild(h("div", { class: "chat-msg assistant" },
      h("div", { class: "chat-role" }, "Assistant"), activity, body));
    current = body;
    currentActivity = activity;
    log.scrollTop = log.scrollHeight;
  }
  function appendStep(text) {
    if (!currentActivity) return;
    const prev = currentActivity.lastElementChild;
    if (prev) prev.classList.remove("live");
    currentActivity.appendChild(h("div", { class: "chat-step live" },
      h("span", { class: "chat-step-dot" }), h("span", null, text)));
    log.scrollTop = log.scrollHeight;
  }
  function scheduleRender() {
    if (raf || !current) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      if (current) current.replaceChildren(renderMarkdown(ctrl.liveRaw));
      log.scrollTop = log.scrollHeight;
    });
  }
  function finishRender() {
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    if (current) {
      current.replaceChildren(renderMarkdown(ctrl.history.length && ctrl.history[ctrl.history.length - 1].role === "assistant"
        ? ctrl.history[ctrl.history.length - 1].content : ""));
      current.classList.remove("streaming");
    }
    if (currentActivity && currentActivity.lastElementChild) currentActivity.lastElementChild.classList.remove("live");
    log.scrollTop = log.scrollHeight;
    setComposerMode();
  }

  // ── proposed-action card (create a draft job from a data path) ────────
  function renderActionCard(action) {
    if (!action) return;
    if (action.kind === "prepare_motion") { renderPrepareMotionCard(action); return; }
    if (action.kind !== "create_job") return;
    const card = h("div", { class: "chat-action" });
    const body = h("div", { class: "chat-action-body" }, h("span", { class: "muted" }, "Preparing proposal…"));
    card.appendChild(h("div", { class: "chat-action-head" },
      icon("sliders", 15),
      h("span", null, "Proposed action: create a ", h("strong", null, action.program), " job")));
    card.appendChild(body);
    log.appendChild(h("div", { class: "chat-msg assistant" },
      h("div", { class: "chat-role" }, "Action"), card));
    log.scrollTop = log.scrollHeight;

    api.assistantPreviewJob(action.program, action.input_path)
      .then(({ spec, inspect }) => fillActionCard(body, action, spec, inspect))
      .catch((e) => { body.replaceChildren(h("span", { class: "chat-action-warn" },
        "Could not prepare the proposal: " + (e && (e.detail || e.message) || "error"))); });
  }
  function fillActionCard(body, action, spec, inspect) {
    body.replaceChildren();
    const facts = [];
    if (inspect) {
      if (!inspect.exists) facts.push("⚠ path not found");
      else {
        if (inspect.kind) facts.push(inspect.kind);
        if (inspect.columns) facts.push(`${inspect.columns.length} columns`);
        if (inspect.volumes != null) facts.push(`${inspect.volumes} volumes`);
        if (inspect.metadata_files != null) facts.push(`${inspect.metadata_files} metadata`);
        if (inspect.box) facts.push(`box ${inspect.box.join("×")}`);
        if (inspect.has_latents) facts.push("has latents");
        if (inspect.has_ctf) facts.push("has CTF");
      }
    }
    body.appendChild(h("div", { class: "chat-action-row" },
      h("span", { class: "muted" }, "Input"),
      h("code", { class: "md-code" }, action.input_path || "(none)"),
      facts.length ? h("span", { class: "chat-action-facts" }, facts.join(" · ")) : null));
    body.appendChild(h("div", { class: "chat-action-row" },
      h("span", { class: "muted" }, "Maps to"),
      h("code", { class: "md-code" }, "--" + (spec.input_arg || "?"))));
    if (spec.missing_required && spec.missing_required.length) {
      body.appendChild(h("div", { class: "chat-action-row" },
        h("span", { class: "muted" }, "You'll set"),
        h("span", { class: "chat-action-missing" },
          spec.missing_required.map((m) => "--" + m).join(", "))));
    }
    const projName = projectName(ctrl.currentChatProject);
    body.appendChild(h("div", { class: "chat-action-row" },
      h("span", { class: "muted" }, "Project"),
      h("span", null, ctrl.currentChatProject ? projName : "none — pick one in the form")));
    const open = h("button", { class: "btn chat-action-open" }, icon("arrowRight", 15), "Open in job builder");
    open.addEventListener("click", () => {
      store.prefillProgram({
        program: action.program, values: spec.args || {},
        projectId: ctrl.currentChatProject || null, projectName: projName,
      });
      body.appendChild(h("div", { class: "chat-action-done muted" },
        "Opened in the job builder (right panel). Complete any required fields and launch it there."));
      open.disabled = true;
    });
    body.appendChild(h("div", { class: "chat-action-actions" }, open,
      h("span", { class: "muted chat-action-note" }, "Draft only — nothing runs until you launch it.")));
  }

  // Confirm card for preparing a run's deformation field (per-domain motion). This
  // runs GPU compute (predict + decode), so it always warns and asks for a GPU
  // before anything runs.
  function renderPrepareMotionCard(action) {
    const card = h("div", { class: "chat-action" });
    const body = h("div", { class: "chat-action-body" }, h("span", { class: "muted" }, "Loading plan…"));
    card.appendChild(h("div", { class: "chat-action-head" }, icon("box", 15),
      h("span", null, "Proposed action: prepare ", h("strong", null, "per-domain motion analysis"))));
    card.appendChild(body);
    log.appendChild(h("div", { class: "chat-msg assistant" },
      h("div", { class: "chat-role" }, "Action"), card));
    log.scrollTop = log.scrollHeight;

    api.assistantPrepareMotionPlan(action.job_id).then(({ plan, gpus }) => {
      body.replaceChildren();
      if (plan.error) { body.appendChild(h("span", { class: "chat-action-warn" }, plan.error)); return; }
      body.appendChild(h("div", { class: "chat-action-warn" }, icon("cpu", 14),
        " Runs on the GPU (" + (plan.predict_needed ? "predict + decode" : "decode") +
        ") — this may take a few minutes."));
      body.appendChild(h("ol", { class: "chat-action-steps" },
        ...(plan.steps || []).map((s) => h("li", null, s))));
      body.appendChild(h("div", { class: "chat-action-row" },
        h("span", { class: "muted" }, "Writes to"),
        h("code", { class: "md-code" }, "decoded_states/deformation/")));

      // GPU picker — only *ask* when several are available (else show the one).
      let gpuValue = (gpus && gpus.length) ? String(gpus[0].index) : "";
      let gpuControl;
      if (gpus && gpus.length > 1) {
        const sel = h("select", { class: "chat-action-gpu" },
          ...gpus.map((g) => h("option", { value: String(g.index) },
            `GPU ${g.index} — ${g.name} (${Math.round(g.mem_free_mb / 1024)} GB free)`)));
        sel.value = gpuValue;
        sel.addEventListener("change", () => { gpuValue = sel.value; });
        gpuControl = sel;
      } else if (gpus && gpus.length === 1) {
        gpuControl = h("span", null, `GPU ${gpus[0].index} — ${gpus[0].name} ` +
          `(${Math.round(gpus[0].mem_free_mb / 1024)} GB free)`);
      } else {
        gpuControl = h("span", { class: "muted" }, "no GPU detected — will use the default device");
      }
      body.appendChild(h("div", { class: "chat-action-row" }, h("span", { class: "muted" }, "GPU"), gpuControl));

      // Live output rendered through the same terminal as job stdout (ANSI colours
      // + carriage-return progress bars update in place).
      const progress = h("div", { class: "chat-action-progress console", style: { display: "none" } });
      const done = h("div");
      const run = h("button", { class: "btn chat-action-open" }, icon("play", 15), "Run");
      const note = h("span", { class: "muted chat-action-note" }, "Nothing runs until you click Run.");
      run.addEventListener("click", () => {
        run.disabled = true; note.textContent = "";
        progress.style.display = "";
        const term = createTerminal(progress);
        term.write("\x1b[2mStarting…\x1b[0m\n");
        api.assistantPrepareMotion({
          jobId: action.job_id, gpu: gpuValue,
          onStatus: (s) => { term.write(s); log.scrollTop = log.scrollHeight; },
          onDone: () => {
            term.write("\n\x1b[1;32m✓ Deformation field ready.\x1b[0m\n");
            done.appendChild(h("div", { class: "chat-action-done" }, icon("check", 14),
              " Ask about the motion again — or regenerate the report — for the per-domain "
              + "hinge/rotation analysis."));
            log.scrollTop = log.scrollHeight;
          },
          onError: (e) => { run.disabled = false; term.write("\n\x1b[1;31mFailed: " + e + "\x1b[0m\n"); },
        });
      });
      body.appendChild(h("div", { class: "chat-action-actions" }, run, note));
      body.appendChild(progress);
      body.appendChild(done);
    }).catch((e) => { body.replaceChildren(h("span", { class: "chat-action-warn" },
      "Could not load the plan: " + (e && (e.detail || e.message) || "error"))); });
  }

  // ── send / hooks ──────────────────────────────────────────────────────
  function send() {
    const text = input.value.trim();
    if (!text || ctrl.streaming) return;
    input.value = "";
    autosize();
    sendMessage(text);   // ctrl updates history + starts the socket turn; onSend renders
  }

  // Hooks the controller calls to drive this view's DOM while it is mounted.
  const hooks = {
    onSend() {
      const last = ctrl.history[ctrl.history.length - 1];
      if (last && last.role === "user") bubble("user", last.content);
      startLiveTurn();
      setComposerMode();
    },
    onStep(text) { appendStep(text); },
    onToken() { scheduleRender(); },
    onAction(rec) { renderActionCard(rec.action); },
    onDone() { finishRender(); },
    onChatsChanged() { refreshChats(); },
  };

  // ── persistence sidebar ───────────────────────────────────────────────
  function projectName(pid) {
    if (!pid) return "No project";
    const p = projects.find((x) => x.id === pid);
    return p ? p.name : "Unknown project";
  }
  async function refreshChats() {
    try { chats = await api.chats(); } catch { chats = []; }
    renderChatList();
  }
  function renderChatList() {
    chatList.replaceChildren();
    const groups = new Map();
    for (const c of chats) {
      const k = c.project_id || "";
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(c);
    }
    if (!chats.length && !dragChatId) {
      chatList.appendChild(h("div", { class: "chat-empty" }, "No chats yet"));
      return;
    }
    // Show every known project (+ "No project") as a drop zone; empty groups are
    // hidden by CSS unless a drag is in progress (chatList gets the .dragging class).
    const keys = new Set(["", ...projects.map((p) => p.id), ...groups.keys()]);
    const ordered = [...keys].sort((a, b) => projectName(a).localeCompare(projectName(b)));
    for (const k of ordered) {
      const items = groups.get(k) || [];
      const group = h("div", {
        class: "chat-group" + (items.length ? "" : " empty-group"),
        ondragover: (e) => { if (!dragChatId) return; e.preventDefault(); e.currentTarget.classList.add("drop-over"); },
        ondragleave: (e) => e.currentTarget.classList.remove("drop-over"),
        ondrop: (e) => {
          e.preventDefault();
          e.currentTarget.classList.remove("drop-over");
          if (dragChatId) moveChatToProject(dragChatId, k);
        },
      },
        h("div", { class: "chat-group-title" }, projectName(k)),
        ...items.map(chatItem));
      chatList.appendChild(group);
    }
  }
  function chatItem(c) {
    const renameBtn = h("button", { class: "chat-item-btn", title: "Rename",
      onclick: (e) => { e.stopPropagation(); renameChat(c); } }, icon("pencil", 14));
    const delBtn = h("button", { class: "chat-item-btn", title: "Delete",
      onclick: (e) => { e.stopPropagation(); removeChat(c); } }, icon("trash", 14));
    return h("div", {
      class: "chat-item" + (c.id === ctrl.currentChatId ? " active" : ""),
      draggable: "true",
      title: "Drag onto a project to move this chat",
      ondragstart: (e) => {
        dragChatId = c.id;
        try { e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", c.id); } catch { /* ignore */ }
        chatList.classList.add("dragging");
      },
      ondragend: () => {
        dragChatId = null;
        chatList.classList.remove("dragging");
        chatList.querySelectorAll(".drop-over").forEach((x) => x.classList.remove("drop-over"));
      },
      onclick: () => selectChat(c.id),
    },
      h("span", { class: "chat-item-title", title: c.title }, c.title),
      h("span", { class: "chat-item-actions" }, renameBtn, delBtn));
  }
  async function moveChatToProject(chatId, projectId) {
    const c = chats.find((x) => x.id === chatId);
    if (c && (c.project_id || "") === (projectId || "")) return;  // already there
    try {
      await api.updateChat(chatId, { project_id: projectId || null });
      if (chatId === ctrl.currentChatId) { ctrl.currentChatProject = projectId || ""; renderAttachRow(); }
      await refreshChats();
    } catch { /* ignore */ }
  }
  async function selectChat(id) {
    if (ctrl.streaming) return;
    try {
      const c = await api.chat(id);
      ctrl.currentChatId = c.id;
      ctrl.currentChatProject = c.project_id || "";
      ctrl.attachedRun = null;
      ctrl.liveActions = [];
      ctrl.history = Array.isArray(c.messages) ? c.messages.slice() : [];
      renderConversation();
      renderChatList();
      renderAttachRow();
    } catch { /* ignore */ }
  }
  function newChat() {
    if (ctrl.streaming) return;
    ctrl.currentChatId = null;
    ctrl.currentChatProject = ctrl.newChatProject || "";
    ctrl.attachedRun = null;
    ctrl.liveActions = [];
    ctrl.history = [];
    renderConversation();
    renderChatList();
    renderAttachRow();
    input.focus();
  }
  async function renameChat(c) {
    const name = window.prompt("Rename chat:", c.title);
    if (name == null) return;
    try { await api.updateChat(c.id, { title: name.trim() || c.title }); await refreshChats(); } catch { /* ignore */ }
  }
  async function removeChat(c) {
    if (!window.confirm(`Delete chat "${c.title}"?`)) return;
    try {
      await api.deleteChat(c.id);
      if (c.id === ctrl.currentChatId) { ctrl.currentChatId = null; ctrl.history = []; ctrl.liveActions = []; renderConversation(); }
      await refreshChats();
    } catch { /* ignore */ }
  }

  function buildProjectSelect() {
    projectSelect.replaceChildren(
      h("option", { value: "" }, "No project"),
      ...projects.map((p) => h("option", { value: p.id, selected: p.id === ctrl.newChatProject }, p.name)));
  }

  // ── init (mount): rebuild from ctrl, then attach hooks ────────────────
  api.projects().then((p) => { projects = p || []; buildProjectSelect(); renderChatList(); }).catch(() => {});
  loadJobs().then(renderAttachRow);
  refreshChats();
  renderConversation();
  renderAttachRow();
  ctrl.hooks = hooks;   // start receiving live updates for this mount

  // ── layout ────────────────────────────────────────────────────────────
  const newBtn = h("button", { class: "btn chat-new", onclick: () => newChat() }, icon("plus", 16), "New chat");
  const sidebar = h("aside", { class: "chat-sidebar" },
    newBtn,
    h("label", { class: "chat-project-row" }, h("span", { class: "muted" }, "New chat in"), projectSelect),
    chatList);
  const header = h("div", { class: "chat-header" },
    h("div", { class: "chat-title" }, icon("chat", 18), h("span", null, "Assistant")), groundBtn);
  const composer = h("div", { class: "chat-composer" }, input, sendBtn);
  const main = h("div", { class: "chat-main" }, header, statusLine, log, attachRow, composer);
  const el = h("div", { class: "view chat-view" }, sidebar, main);

  // Detach on route change: stop touching the DOM, but keep the socket + the
  // turn running in the background so the answer is intact when we return.
  return { el, cleanup: () => { if (ctrl.hooks === hooks) ctrl.hooks = null; } };
}
