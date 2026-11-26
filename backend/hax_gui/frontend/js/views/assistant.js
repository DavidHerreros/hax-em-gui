// AI assistant chat panel: a chat-history sidebar (new / rename / delete, grouped
// by project) beside a streaming conversation with a live activity trace, Markdown
// rendering, and a Stop button. Only routed when /api/assistant/status reports the
// optional hax-em-assistant package is installed (see app.js).
//
// The conversation + WebSocket + streaming buffers live in a module-level
// controller (`ctrl`) that OUTLIVES the route view, so a question keeps running in
// the background when you leave the page: on return the view rebuilds from `ctrl`
// and you see the ongoing answering process — or the finished answer.
import { h, clear, append } from "../h.js";
import { icon } from "../icons.js";
import { errText, toastError } from "../toast.js";
import { confirmModal } from "./confirmModal.js";
import { assistantChat, api } from "../api.js";
import { renderMarkdown } from "../markdown.js";
import { navigate, currentPath } from "../router.js";
import { store } from "../store.js";
import { createTerminal } from "./terminal.js";
import { runBrowser, folderName } from "./runBrowser.js";
import { jobLabelChips } from "./labels.js";
import { jobTimeEl } from "./jobs.js";
import { rememberWorkflowToOpen } from "./workflows.js";
import { loadCommands, customCommands, findCommand, expandCommand, onCommandsChange,
         slashMenu, openCommandEditor } from "./assistantCommands.js";
import { setFloating } from "./assistantFloat.js";

const GREETING =
  "Hi! I can help you pick programs and parameters, explain options from the docs, "
  + "diagnose a failed run's log, and reason about results. Ask me anything about your hax workflows.";

// ── persistent controller (survives view unmount) ───────────────────────────
const ctrl = {
  history: [],             // {role, content} — the active conversation
  currentChatId: null,     // persisted chat id (null = unsaved new chat)
  currentChatProject: "",  // project_id of the active chat (scopes the run picker)
  newChatProject: "",      // project_id for the next new chat
  attachedRun: null,       // {job_id,label,has_*} attached-run context (primary: full artifacts)
  extraRuns: [],           // up to 2 more runs: measured facts + parameter diff only
  grounded: true,
  // live turn state (an in-flight answer)
  streaming: false,
  turn: 0,                 // client turn id (echoed on the server's done)
  liveRaw: "",             // partial assistant answer accumulated so far
  liveSteps: [],           // activity-trace step texts for the current turn
  liveActions: [],         // proposed-action records for this turn
  liveQuestion: null,      // a question the assistant is waiting on (ask-and-wait)
  stopped: false,
  liveError: null,         // transient transport/model error — shown, never persisted
  socket: null,            // one WebSocket for the app's lifetime (reused per turn)
  hooks: null,             // set by the mounted view; null while unmounted
  turnNoAct: false,        // this turn came from a command that may not act: its proposals stay drafts
  unread: false,           // an answer finished while no view was mounted
};
// What the floating chat's pill shows while the chat is minimised.
const watchers = new Set();
function changed() { for (const fn of watchers) fn(); }
export function onAssistantActivity(fn) { watchers.add(fn); return () => watchers.delete(fn); }
export function assistantActivity() {
  return { streaming: ctrl.streaming, unread: ctrl.unread,
           pending: (ctrl.liveQuestion ? 1 : 0) + ctrl.liveActions.filter((r) => !r.resolved).length };
}
function resolved(rec) { if (rec && !rec.resolved) { rec.resolved = true; changed(); } }

const TABS = [["/jobs", "Project"], ["/project", "Project"], ["/metrics", "Metrics"], ["/viewer", "3D viewer"],
              ["/workflows", "Workflows"], ["/compare", "Compare"], ["/settings", "Settings"], ["/assistant", ""]];
// The tab the user is on, and the run it shows when the route names one.
function describeRoute() {
  const path = currentPath();
  const hit = TABS.find(([p]) => path.startsWith(p));
  const m = /^\/jobs\/([^/]+)/.exec(path);
  return { tab: hit ? hit[1] : "Home", jobId: m ? decodeURIComponent(m[1]) : null };
}
// Workflow-card state (preview, creation) keyed by its action object, which ctrl.liveActions
// keeps across re-renders: a re-rendered card shows the same state and never creates twice.
const wfCards = new WeakMap();

function ensureSocket() {
  if (ctrl.socket) return ctrl.socket;
  ctrl.socket = assistantChat({
    onStatus: (text) => {
      if (!ctrl.streaming) return;
      ctrl.liveSteps.push(text);
      if (ctrl.hooks) ctrl.hooks.onStep(text);
    },
    onAction: (action) => {
      if (!ctrl.streaming) return;   // a late action from a stopped turn
      const rec = { action, noAct: ctrl.turnNoAct };
      ctrl.liveActions.push(rec);
      changed();
      if (ctrl.hooks) ctrl.hooks.onAction(rec);
    },
    onQuestion: (q) => {
      if (!ctrl.streaming) return;   // a late question from a stopped turn
      ctrl.liveQuestion = q;
      changed();
      if (ctrl.hooks) ctrl.hooks.onQuestion(q);
    },
    onToken: (t) => {
      if (!ctrl.streaming) return;
      ctrl.liveRaw += t;
      if (ctrl.hooks) ctrl.hooks.onToken();
    },
    // Ignore a done that belongs to an earlier (stopped) turn: after
    // stop-then-ask, the cancelled turn's done can land while the new
    // question is already streaming and must not complete it.
    onDone: (turn) => {
      if (ctrl.streaming && (turn == null || turn === ctrl.turn)) completeTurn();
    },
    onError: (err) => {
      if (!ctrl.streaming) return;
      // Shown in the transcript but NEVER pushed into history: a transport
      // error used to become a permanent assistant turn, re-sent to the model
      // as context on every later message in the chat.
      ctrl.liveError = String(err);
      completeTurn();
    },
    onClose: () => {
      // The socket dropped (server restart, network). The turn is lost
      // server-side; unlock the composer instead of staying on "Stop" forever.
      if (!ctrl.streaming) return;
      ctrl.liveError = "connection lost — the server went away mid-answer";
      completeTurn();
    },
  });
  return ctrl.socket;
}

// `shown` is what the bubble displays when the text is a command's expanded prompt.
function sendMessage(text, { shown, noAct } = {}) {
  ctrl.history.push(shown ? { role: "user", content: text, shown } : { role: "user", content: text });
  ctrl.turnNoAct = Boolean(noAct);
  ctrl.streaming = true;
  ctrl.turn = (ctrl.turn || 0) + 1;
  ctrl.liveRaw = "";
  ctrl.liveSteps = [];
  ctrl.liveActions = [];
  ctrl.stopped = false;
  ctrl.liveError = null;
  const payload = { messages: ctrl.history.map(({ role, content }) => ({ role, content })),
                    grounded: ctrl.grounded, turn: ctrl.turn };
  if (ctrl.attachedRun) payload.run_job_id = ctrl.attachedRun.job_id;
  if (ctrl.extraRuns.length) payload.extra_run_job_ids = ctrl.extraRuns.map((r) => r.job_id);
  ensureSocket().send(payload);
  if (ctrl.hooks) ctrl.hooks.onSend();
  changed();
}

function completeTurn() {
  const pushed = Boolean(ctrl.liveRaw);
  if (pushed) ctrl.history.push({ role: "assistant", content: ctrl.liveRaw });
  else if (!ctrl.stopped && ctrl.history.length &&
           ctrl.history[ctrl.history.length - 1].role === "user") ctrl.history.pop();
  ctrl.streaming = false;
  ctrl.liveRaw = "";
  ctrl.liveQuestion = null;
  if (pushed && !ctrl.hooks) ctrl.unread = true;
  changed();
  // `pushed` tells the view whether the live bubble has an answer to show:
  // rendering "the last assistant message" unconditionally used to duplicate
  // the PREVIOUS answer into the new bubble on an empty turn.
  if (ctrl.hooks) ctrl.hooks.onDone(pushed);
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
  return u ? (u.shown || u.content).slice(0, 60) : "New chat";
}
// Serialised: two overlapping persists used to both see currentChatId === null
// and both POST a new chat (one duplicate, one empty); and re-reading the
// globals after an await could write chat A's messages into chat B after a
// switch. Each persist snapshots the conversation (array identity marks a
// switch — selectChat/newChat replace ctrl.history wholesale) and runs after
// the previous one has settled.
let persistChain = Promise.resolve();
function persistCurrent() {
  const hist = ctrl.history;
  persistChain = persistChain.then(async () => {
    if (!hist.length || ctrl.history !== hist) return;   // empty, or switched away
    if (!ctrl.currentChatId) {
      const c = await api.createChat(ctrl.newChatProject || null, deriveTitle());
      if (ctrl.history !== hist) return;   // switched while creating — don't adopt/write
      ctrl.currentChatId = c.id;
    }
    await api.updateChat(ctrl.currentChatId, { messages: hist });
    if (ctrl.hooks && ctrl.hooks.onChatsChanged) ctrl.hooks.onChatsChanged();
  }).catch(() => { /* persistence is best-effort */ });
  return persistChain;
}

// ── the route view (rebuilds from ctrl on mount; detaches on unmount) ────────
// `compact` is the floating window: no chat list, a line saying what the user is looking
// at, and `headerExtra` (the window buttons) in the header, which doubles as drag handle.
export function AssistantView(_params, { compact = false, headerExtra = [] } = {}) {
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
    placeholder: "Ask anything, or type / for commands", "aria-label": "Message",
  });
  const sendBtn = h("button", { class: "chat-send" });
  sendBtn.addEventListener("click", () => { if (ctrl.streaming) stopTurn(); else send(); });
  function setComposerMode() {
    sendBtn.replaceChildren(icon(ctrl.streaming ? "stop" : "arrowUp", ctrl.streaming ? 14 : 17));
    sendBtn.title = ctrl.streaming ? "Stop" : "Send";
    sendBtn.setAttribute("aria-label", sendBtn.title);
    sendBtn.classList.toggle("danger", ctrl.streaming);
  }

  const groundBtn = h("button", {
    class: "chip",
    title: "Ground answers in the hax docs + program schema",
    onclick: () => { ctrl.grounded = !ctrl.grounded; groundBtn.classList.toggle("active", ctrl.grounded); },
  }, "docs grounding");
  groundBtn.classList.toggle("active", ctrl.grounded);

  // Autonomy (also `/autonomy [level]` in the composer): may the assistant act on jobs?
  let autonomyLevel = "";
  const autonomySeg = h("div", {
    class: "chat-seg", role: "group", "aria-label": "Autonomy", style: { display: "none" },
    title: "Autonomy: may the assistant launch, stop or relaunch jobs? (also /autonomy in the chat)",
  }, ...[["advisory", "Advisory"], ["ask", compact ? "Ask" : "Ask first"], ["auto", "Auto"]].map(([level, label]) =>
    h("button", { "data-level": level, onclick: () => { if (level !== autonomyLevel) autonomyCommand(level); } }, label)));
  function showAutonomy(level) {
    autonomyLevel = level;
    for (const b of autonomySeg.children) {
      b.classList.toggle("active", b.dataset.level === level);
      b.setAttribute("aria-pressed", String(b.dataset.level === level));
    }
  }
  let workspaceError = null;
  const wsWarn = h("span", { class: "chat-action-warn" });

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
  const statusLine = h("div", { class: "chat-status" }, statusText, reindexBtn, wsWarn);
  function showStatus(s) {
    statusText.textContent = s.available
      ? `model: ${s.model}  ·  docs: ${s.docs_grounding ? "on" : "off"}  ·  ${s.programs} programs, ${s.doc_chunks} doc chunks`
      : "assistant unavailable";
    reindexBtn.style.display = s.available ? "" : "none";
    autonomySeg.style.display = s.available && s.autonomy ? "" : "none";
    if (s.autonomy) showAutonomy(s.autonomy);
    workspaceError = (s.available && s.workspace_error) || null;
    wsWarn.textContent = workspaceError ? "⚠ assistant actions off" : "";
    wsWarn.title = workspaceError || "";
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
    renderContext();
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
      // Extra runs (cross-run reasoning): facts + parameter diff, no artifacts.
      ctrl.extraRuns.forEach((er, i) => {
        attachRow.appendChild(h("div", { class: "chat-attach-chip extra", title: "Compared against the primary run: the model sees its measured facts and the flags that differ" },
          icon("grid", 13),
          h("span", { class: "chat-attach-name" }, er.label),
          h("button", { class: "chat-item-btn", title: "Detach", onclick: () => { ctrl.extraRuns.splice(i, 1); renderAttachRow(); } }, icon("x", 13))));
      });
      if (ctrl.extraRuns.length < 2) {
        attachRow.appendChild(h("button", { class: "chat-attach-btn",
          title: "Attach another run to compare against — the model gets its measured facts and the parameter differences",
          onclick: openRunPicker }, icon("plus", 13), "Compare with…"));
      }
    }
  }
  // Floating window only: the tab behind it, and the run it shows when the route names one.
  const contextRow = h("div", { class: "chat-context" });
  let askedJobs = null;
  function renderContext() {
    if (!compact) return;
    const here = describeRoute();
    const job = here.jobId && jobsCache.find((j) => j.id === here.jobId);
    if (here.jobId && !job && askedJobs !== here.jobId) { askedJobs = here.jobId; loadJobs().then(renderContext); }
    const attached = [ctrl.attachedRun, ...ctrl.extraRuns].some((r) => r && r.job_id === here.jobId);
    clear(contextRow);
    append(contextRow, [icon("eye", 13),
      h("span", null, "Looking at ", h("strong", null, here.tab + (job ? ` · ${job.name || job.program}` : ""))),
      here.jobId && !attached ? h("button", { class: "chat-link", onclick: () => attachJob(here.jobId) }, "Attach this run") : null]);
  }
  async function attachJob(id) {
    if (!id) { ctrl.attachedRun = null; ctrl.extraRuns = []; renderAttachRow(); return; }
    let preview;
    try { preview = await api.assistantRunPreview(id); }
    catch { preview = { job_id: id, label: "run", has_metrics: false, has_latents: false, has_volumes: false }; }
    if (ctrl.attachedRun && ctrl.attachedRun.job_id !== id) {
      // A second/third pick becomes a comparison run rather than replacing the primary.
      if (!ctrl.extraRuns.some((r) => r.job_id === id) && ctrl.extraRuns.length < 2) ctrl.extraRuns.push(preview);
    } else {
      ctrl.attachedRun = preview;
    }
    renderAttachRow();
  }
  // Detaching the primary promotes the first comparison run (its artifacts load next turn).
  function detachRun() { ctrl.attachedRun = ctrl.extraRuns.shift() || null; renderAttachRow(); }
  async function downloadReport() {
    if (!ctrl.attachedRun || reportBusy) return;
    reportBusy = true; renderAttachRow();
    try {
      await api.assistantRunReport(ctrl.attachedRun.job_id, true);
    } catch (e) {
      toastError(e, "Report generation failed.");
    } finally {
      reportBusy = false; renderAttachRow();
    }
  }

  const chatList = h("div", { class: "chat-list" });
  const titleEl = h("span", null, "New chat");
  const projectSelect = h("select", {
    class: "chat-project-select",
    onchange: (e) => {
      ctrl.newChatProject = e.target.value;
      // an unsaved chat follows the picker's project scope immediately
      if (!ctrl.currentChatId) { ctrl.currentChatProject = ctrl.newChatProject; renderAttachRow(); }
    },
  });

  // ── conversation rendering ────────────────────────────────────────────
  function bubble(role, text, shown) {
    const body = h("div", { class: "chat-body" });
    const cmd = shown && /^(\/\S+)\s*([\s\S]*)$/.exec(shown);
    if (cmd) { body.title = text; body.append(h("code", { class: "chat-cmd" }, cmd[1]), cmd[2] ? " " + cmd[2] : ""); }
    else if (role === "user") body.textContent = text;
    else body.replaceChildren(renderMarkdown(text));
    log.appendChild(h("div", { class: `chat-msg ${role}` }, body));
    log.scrollTop = log.scrollHeight;
    return body;
  }
  // Rebuild the whole log from ctrl: history, the in-flight turn (if streaming),
  // and any proposed-action cards. Called on mount and on chat switch.
  function renderConversation() {
    log.replaceChildren();
    current = currentActivity = questionCard = null;
    if (!ctrl.history.length && !ctrl.streaming) { bubble("assistant", GREETING); }
    else for (const m of ctrl.history) bubble(m.role, m.content, m.shown);
    if (ctrl.streaming) {
      startLiveTurn();
      for (const s of ctrl.liveSteps) appendStep(s);
      if (ctrl.liveRaw) current.replaceChildren(renderMarkdown(ctrl.liveRaw));
    }
    for (const rec of ctrl.liveActions) renderActionCard(rec.action, rec);
    if (ctrl.streaming && ctrl.liveQuestion) renderQuestionCard(ctrl.liveQuestion);
    log.scrollTop = log.scrollHeight;
    setComposerMode();
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }
  // "/" menu: Enter runs the command, Tab (or a command that takes a value) fills it in.
  const editCommands = (select) => openCommandEditor({
    select, lastMessage: (ctrl.history.filter((m) => m.role === "user" && !m.shown).pop() || {}).content || "",
    onTry: (draft) => runCustom(draft, ""),
  });
  const menu = slashMenu({
    autonomy: () => autonomyLevel,
    onNew: () => { menu.close(); editCommands(); },
    onManage: () => { menu.close(); editCommands((customCommands()[0] || {}).name); },
    onPick: (item, fill) => {
      menu.close();
      input.value = "/" + item.name + (fill || item.arg ? " " : "");
      if (fill || item.arg) input.focus(); else send();
    },
  });
  input.addEventListener("input", () => { autosize(); menu.update(input.value); });
  input.addEventListener("blur", () => menu.close());
  input.addEventListener("keydown", (e) => {
    if (menu.key(e)) return;
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });

  // ── live turn DOM ─────────────────────────────────────────────────────
  function startLiveTurn() {
    const activity = h("details", { class: "chat-activity" },
      h("summary", { class: "chat-step live" }, h("span", { class: "chat-step-dot" }), h("span", null, "Thinking…")),
      h("div", { class: "chat-steps" }));
    const body = h("div", { class: "chat-body streaming" });
    log.appendChild(h("div", { class: "chat-msg assistant" }, activity, body));
    current = body;
    currentActivity = activity;
    log.scrollTop = log.scrollHeight;
  }
  function appendStep(text) {
    if (!currentActivity) return;
    currentActivity.firstElementChild.lastElementChild.textContent = text;
    currentActivity.lastElementChild.appendChild(h("div", { class: "chat-step" },
      h("span", { class: "chat-step-dot" }), h("span", null, text)));
    log.scrollTop = log.scrollHeight;
  }
  // Re-parsing the whole accumulated answer on every animation frame was
  // O(n²) across a long reply and visibly janked. Throttle to one full render
  // per RENDER_MS; finishRender always does a final exact pass.
  const RENDER_MS = 150;
  let lastRender = 0;
  function scheduleRender() {
    if (raf || !current) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const now = performance.now();
      if (now - lastRender < RENDER_MS) { scheduleLate(); return; }
      lastRender = now;
      if (current) current.replaceChildren(renderMarkdown(ctrl.liveRaw));
      log.scrollTop = log.scrollHeight;
    });
  }
  let lateTimer = 0;
  function scheduleLate() {
    // Tokens stopped arriving before the throttle window closed: make sure the
    // tail still gets painted.
    if (lateTimer) return;
    lateTimer = setTimeout(() => { lateTimer = 0; scheduleRender(); }, RENDER_MS);
  }
  function finishRender(pushed) {
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    if (lateTimer) { clearTimeout(lateTimer); lateTimer = 0; }
    if (current) {
      // Only show an answer this turn actually produced. Reading "the last
      // assistant message" unconditionally re-rendered the PREVIOUS answer
      // into the new bubble whenever a turn ended empty.
      const content = pushed && ctrl.history.length &&
        ctrl.history[ctrl.history.length - 1].role === "assistant"
        ? ctrl.history[ctrl.history.length - 1].content : "";
      current.replaceChildren(renderMarkdown(content));
      if (ctrl.liveError) {
        current.appendChild(h("div", { class: "chat-error muted" }, `⚠ ${ctrl.liveError}`));
        ctrl.liveError = null;
      }
      current.classList.remove("streaming");
    }
    if (currentActivity) {
      const summary = currentActivity.firstElementChild, n = currentActivity.lastElementChild.childElementCount;
      summary.classList.remove("live");
      if (n > 1) summary.lastElementChild.textContent = `${n} steps`;
      else if (!n) currentActivity.remove();
      currentActivity = null;
    }
    dismissQuestion();   // the turn ended (or was stopped): nothing waits on it any more
    log.scrollTop = log.scrollHeight;
    setComposerMode();
  }

  // ── cards ─────────────────────────────────────────────────────────────
  // The badge says what the card wants from you: "proposal" (nothing runs from here),
  // "ask" (waits for your OK), "done" (already acted), "blocked", "muted" (dismissed).
  const badge = (kind, text) => h("span", { class: `chat-badge ${kind}` }, text);
  const cardHead = (iconName, kind, text, ...title) => h("div", { class: "chat-action-head" },
    icon(iconName, 15), h("span", { class: "chat-action-title" }, ...title), badge(kind, text));

  // ── a question the assistant is waiting on (ask-and-wait) ─────────────
  // The answer goes back as an {answer:{id, value}} frame, which does not stop
  // the turn; Skip sends {id, cancel:true}; Stop in the composer ends the turn.
  let questionCard = null;
  function dismissQuestion() {
    if (questionCard) { questionCard.remove(); questionCard = null; }
  }
  function renderQuestionCard(q) {
    dismissQuestion();
    const controls = h("div", { class: "chat-action-actions" });
    const reply = (payload, shown) => {
      ctrl.liveQuestion = null;
      changed();
      ensureSocket().answer({ id: q.id, ...payload });
      controls.replaceChildren(h("span", { class: "muted chat-action-note" }, shown));
    };
    const choices = q.kind === "confirm" ? ["yes", "no"] : (q.kind === "choice" ? (q.choices || []) : []);
    if (choices.length) {
      for (const c of choices) {
        controls.appendChild(h("button", {
          class: "btn small" + (c === q.default ? "" : " secondary"),
          onclick: () => reply({ value: c }, `Answered: ${c}`),
        }, c));
      }
    } else {
      const field = h("input", {
        class: "chat-action-gpu", style: { minWidth: "260px" }, value: q.default ?? "",
        placeholder: q.kind === "path" ? "a path on the server" : "your answer",
      });
      const submit = () => reply({ value: field.value }, `Answered: ${field.value || q.default || ""}`);
      field.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } });
      controls.appendChild(field);
      controls.appendChild(h("button", { class: "btn small", onclick: submit }, "Answer"));
      setTimeout(() => field.focus(), 0);
    }
    controls.appendChild(h("button", { class: "btn small ghost",
      onclick: () => reply({ cancel: true }, "Skipped.") }, "Skip"));
    const card = h("div", { class: "chat-action" },
      cardHead("chat", "ask", "Needs your answer", "The assistant asks"),
      h("div", { class: "chat-action-body" },
        h("div", { style: { whiteSpace: "pre-wrap" } }, q.text),
        q.error ? h("div", { class: "chat-action-warn" }, q.error) : null,
        controls));
    questionCard = h("div", { class: "chat-msg assistant" }, card);
    log.appendChild(questionCard);
    log.scrollTop = log.scrollHeight;
  }

  // ── client-side commands (never sent to the model) ────────────────────
  function notice(text) {
    log.appendChild(h("div", { class: "chat-msg assistant" },
      h("div", { class: "chat-step" }, h("span", { class: "chat-step-dot" }), h("span", null, text))));
    log.scrollTop = log.scrollHeight;
  }
  async function autonomyCommand(level) {
    try {
      const a = level ? await api.setAssistantAutonomy(level) : await api.assistantAutonomy();
      showAutonomy(a.level);
      const others = Object.keys(a.levels || {}).filter((l) => l !== a.level);
      notice(`autonomy: ${a.level} — ${a.description}`
        + (level ? "" : `  (change with /autonomy ${others.join(" | ")})`)
        + (a.env_override ? "  [HAX_ASSISTANT_AUTONOMY overrides the saved setting]" : "")
        + (workspaceError && a.level !== "advisory" ? `  ⚠ actions are refused: ${workspaceError}` : ""));
    } catch (e) {
      notice(`autonomy: ${errText(e)}`);
      api.assistantAutonomy().then((a) => showAutonomy(a.level)).catch(() => {});
    }
  }

  // ── proposed-action card (create a draft job from a data path) ────────
  function renderActionCard(action, rec) {
    if (!action) return;
    if (action.kind === "prepare_motion") { renderPrepareMotionCard(action, rec); return; }
    if (action.kind === "workflow") { renderWorkflowCard(action, rec); return; }
    if (action.kind !== "create_job") return;
    const card = h("div", { class: "chat-action" });
    const body = h("div", { class: "chat-action-body" }, h("span", { class: "muted" }, "Preparing proposal…"));
    card.appendChild(cardHead("sliders", "proposal", "Proposal", "Create a ", h("strong", null, action.program), " job"));
    card.appendChild(body);
    log.appendChild(h("div", { class: "chat-msg assistant" }, card));
    log.scrollTop = log.scrollHeight;

    api.assistantPreviewJob(action.program, action.input_path)
      .then(({ spec, inspect }) => fillActionCard(body, action, spec, inspect, rec))
      .catch((e) => { body.replaceChildren(h("span", { class: "chat-action-warn" },
        "Could not prepare the proposal: " + (e && (e.detail || e.message) || "error"))); });
  }
  function fillActionCard(body, action, spec, inspect, rec) {
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
      resolved(rec);
    });
    body.appendChild(h("div", { class: "chat-action-actions" }, open,
      h("span", { class: "muted chat-action-note" }, "Draft only — nothing runs until you launch it.")));
  }

  // ── proposed workflow: re-planned on the server, run by the workflow engine ──
  // What the card may do follows the autonomy decision in the preview: propose -> draft
  // only; confirm -> draft or "Create and run"; proceed -> created and started at once.
  const WF_MARK = { pass: "✓", warn: "⚠", fail: "✗" };
  const argText = (v) => (v === true ? "on" : v === false ? "off" : String(v));
  function wfErrText(e) {
    const d = e && e.detail;
    if (d && typeof d === "object" && d.error) {
      return [d.error, ...(d.failed_checks || []).map((c) => `${c.name}: ${c.message}`), ...(d.errors || [])].join(" · ");
    }
    return errText(e);
  }
  function openInBuilder(id) { rememberWorkflowToOpen(id, { first: true }); navigate("/workflows"); }

  function renderWorkflowCard(action, rec) {
    let st = wfCards.get(action);
    if (!st) {
      st = { request: (action.form && action.form.raw) || "", projectId: ctrl.currentChatProject || null,
             created: null, creating: null, error: null, autoTried: false, dismissed: false, rec };
      st.preview = api.assistantPreviewWorkflow({ request: st.request, projectId: st.projectId });
      wfCards.set(action, st);
    }
    const label = String(action.template || "workflow").replace(/_run$/, " (existing run)").replace(/_/g, " ");
    const body = h("div", { class: "chat-action-body" }, h("span", { class: "muted" }, "Checking the plan…"));
    log.appendChild(h("div", { class: "chat-msg assistant" },
      h("div", { class: "chat-action" }, cardHead("workflow", "muted", "Checking", "Workflow: ", h("strong", null, label)), body)));
    log.scrollTop = log.scrollHeight;
    st.preview.then((p) => { fillWorkflowCard(body, st, p); log.scrollTop = log.scrollHeight; })
      .catch((e) => body.replaceChildren(h("span", { class: "chat-action-warn" },
        "Could not prepare the workflow: " + wfErrText(e))));
  }

  function fillWorkflowCard(body, st, p) {
    body.replaceChildren(h("div", { class: "muted" }, p.plan.description));
    const inheritedNote = (s) => {
      const names = Object.keys(s.inherited);
      return names.length ? h("div", { class: "chat-wf-why" }, `${names.map((n) => "--" + n).join(", ")}: taken from `
        + `${Object.values(s.inherited)[0]}, as the workflow builder does when wiring its network`) : null;
    };
    body.appendChild(h("ol", { class: "chat-action-steps" }, ...p.steps.map((s) => h("li", null,
      h("strong", null, s.title), s.note ? h("span", { class: "muted" }, ` — ${s.note}`) : null,
      ...Object.entries(s.args).map(([flag, v]) => h("div", { class: "chat-wf-arg" },
        h("code", { class: "md-code" }, `--${flag} ${argText(v)}`),
        s.reasons[flag] ? h("div", { class: "chat-wf-why" }, s.reasons[flag]) : null)),
      ...Object.entries(s.wired).map(([flag, src]) => h("div", { class: "chat-wf-arg" },
        h("code", { class: "md-code" }, `--${flag}`), h("div", { class: "chat-wf-why" }, `← the output of ${src}`))),
      inheritedNote(s)))));
    for (const o of p.omitted) {
      body.appendChild(h("div", { class: "chat-wf-arg muted" }, `${o.index}. ${o.title} — ${o.reason}`));
    }
    const shown = p.plan.checks.filter((c) => c.status !== "pass");
    const passed = p.plan.checks.filter((c) => c.status === "pass");
    const checkRow = (c) => h("div", { class: `chat-wf-check ${c.status}` },
      h("span", { class: "chat-wf-mark" }, WF_MARK[c.status] || "·"), h("span", null, `${c.name}: ${c.message}`));
    body.appendChild(h("div", { class: "chat-wf-checks" }, ...shown.map(checkRow),
      passed.length ? h("details", null, h("summary", { class: "muted" }, `${passed.length} pre-flight checks passed`),
        ...passed.map(checkRow)) : null));
    for (const e of p.errors) body.appendChild(h("div", { class: "chat-action-warn" }, e));
    body.appendChild(h("div", { class: "chat-action-row" }, h("span", { class: "muted" }, "Project"),
      h("span", null, p.project.new ? `a new project “${p.project.name}”` : p.project.name || p.project.id)));
    const actions = h("div", { class: "chat-action-actions" });
    body.appendChild(actions);
    renderWorkflowActions(actions, st, p);
  }

  function renderWorkflowActions(el, st, p) {
    // A command that may not act keeps its proposals as drafts, whatever the autonomy.
    const decision = st.rec && st.rec.noAct ? "propose" : p.autonomy.decision;
    const blocked = !p.ok;
    // Once answered the card folds to its head and this line, so the conversation stays short.
    const card = el.closest(".chat-action");
    if (card) {
      const c = st.created;
      const [kind, text] = c ? (!c.started ? ["proposal", "Draft saved"] : c.auto ? ["done", "Started by the assistant"] : ["done", "Started"])
        : st.dismissed ? ["muted", "Dismissed"] : blocked ? ["blocked", "Blocked"]
        : decision === "confirm" ? ["ask", "Needs your OK"] : decision === "proceed" ? ["done", "Auto"] : ["proposal", "Proposal"];
      card.querySelector(".chat-badge").replaceWith(badge(kind, text));
      card.classList.toggle("folded", Boolean(c || st.dismissed));
    }
    if (st.created || st.dismissed || blocked) resolved(st.rec);
    if (st.dismissed) {
      el.replaceChildren(h("span", { class: "muted" }, "Dismissed. Nothing was changed."),
        h("button", { class: "chat-link", onclick: () => { st.dismissed = false; renderWorkflowActions(el, st, p); } }, "Show again"));
      return;
    }
    if (st.created) {
      const c = st.created;
      const text = !c.started
        ? (c.run_error ? `Saved “${c.name}” as a draft, but it could not start: ${c.run_error}`
                       : `Saved “${c.name}” as a draft. Nothing runs until you run it.`)
        : c.auto ? `Autonomy is auto, so the assistant created and started “${c.name}” on its own. `
                   + "Its steps run one after another in the project; the training is monitored."
                 : `Created and started “${c.name}”. Its steps run one after another in the project; `
                   + "the training is monitored.";
      el.replaceChildren(h("div", { class: c.started ? "chat-action-done" : "muted" }, icon(c.started ? "play" : "check", 14), text),
        h("button", { class: "btn small chat-action-open", onclick: () => openInBuilder(c.workflow_id) },
          icon("workflow", 14), "Open in the workflow builder"));
      return;
    }
    if (st.creating) {
      el.replaceChildren(h("span", { class: "muted" }, st.creatingText));
      st.creating.then(() => renderWorkflowActions(el, st, p));
      return;
    }
    const go = decision === "confirm" ? h("button", { class: "btn chat-action-open", disabled: blocked ? "true" : undefined,
      onclick: () => createWorkflow(el, st, p, true, false) }, icon("play", 15), "Create and run") : null;
    const draft = h("button", { class: "btn chat-action-open" + (go ? " secondary" : ""), disabled: blocked ? "true" : undefined,
      onclick: () => createWorkflow(el, st, p, false, false) }, icon("workflow", 15), go ? "Save as draft" : "Open as draft");
    const dismiss = h("button", { class: "btn ghost chat-action-open",
      onclick: () => { st.dismissed = true; renderWorkflowActions(el, st, p); } }, "Dismiss");
    const n = p.steps.length;
    const note = blocked ? "Failed checks: nothing can be created until they pass."
      : decision === "propose" ? "Nothing runs until you run it."
      : decision === "confirm" ? `Starts ${n} job${n === 1 ? "" : "s"}${n > 1 ? ", one after the other" : ""}`
      : "";
    el.replaceChildren(go || "", draft, dismiss, note ? h("span", { class: "muted chat-action-note" }, note) : "",
      st.error ? h("div", { class: "chat-action-warn" }, st.error) : "");
    // auto: act once, and say so on the card; a failure falls back to the buttons above
    if (decision === "proceed" && !blocked && !st.autoTried) { st.autoTried = true; createWorkflow(el, st, p, true, true); }
  }

  function createWorkflow(el, st, p, run, auto) {
    if (st.creating || st.created) return;          // one workflow per card
    st.error = null;
    st.creatingText = !run ? "Saving the draft…"
      : auto ? "Autonomy is auto: creating and starting the workflow…" : "Creating and starting the workflow…";
    st.creating = api.assistantCreateWorkflow({ request: st.request, projectId: st.projectId, run, auto,
                                                fingerprint: p.fingerprint })
      .then((res) => {
        st.created = { ...res, auto };
        if (!run && ctrl.hooks === hooks) openInBuilder(res.workflow_id);   // only while this view is up
      })
      .catch((e) => { st.error = wfErrText(e); })
      .finally(() => { st.creating = null; });
    renderWorkflowActions(el, st, p);
  }

  // Confirm card for preparing a run's deformation field (per-domain motion). This
  // runs GPU compute (predict + decode), so it always warns and asks for a GPU
  // before anything runs.
  function renderPrepareMotionCard(action, rec) {
    const card = h("div", { class: "chat-action" });
    const body = h("div", { class: "chat-action-body" }, h("span", { class: "muted" }, "Loading plan…"));
    card.appendChild(cardHead("box", "ask", "Needs your OK", "Prepare ", h("strong", null, "per-domain motion analysis")));
    card.appendChild(body);
    log.appendChild(h("div", { class: "chat-msg assistant" }, card));
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
        resolved(rec);
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
  async function runCustom(cmd, arg) {
    const here = describeRoute();
    if (cmd.attach_run && !ctrl.attachedRun && here.jobId) await attachJob(here.jobId);
    const text = expandCommand(cmd, {
      run: ctrl.attachedRun && ctrl.attachedRun.label, tab: here.tab, text: arg,
      project: ctrl.currentChatProject ? projectName(ctrl.currentChatProject) : "",
    });
    if (ctrl.streaming) return;
    sendMessage(text, { shown: `/${cmd.name}${arg ? " " + arg : ""}`, noAct: !cmd.may_act });
  }
  // True when `name` was a command and has been handled: the model never sees the line.
  function runCommand(name, arg) {
    if (name === "autonomy") autonomyCommand(arg);
    else if (name === "attach") openRunPicker();
    else if (name === "report") { if (ctrl.attachedRun) downloadReport(); else notice("report: attach a run first (/attach)"); }
    else if (name === "float") { setFloating(!compact); if (compact) navigate("/assistant"); }
    else if (name === "new") newChat();
    else if (name === "commands") editCommands();
    else {
      const cmd = findCommand(name);
      if (!cmd || ctrl.streaming) return false;
      runCustom(cmd, arg);
    }
    return true;
  }
  function send() {
    const text = input.value.trim();
    const cmd = /^\/([a-z0-9-]+)(?:\s+([\s\S]*))?$/i.exec(text);
    if (cmd && runCommand(cmd[1].toLowerCase(), (cmd[2] || "").trim())) {
      input.value = "";
      autosize();
      menu.close();
      return;
    }
    if (!text || ctrl.streaming) return;
    input.value = "";
    autosize();
    sendMessage(text);   // ctrl updates history + starts the socket turn; onSend renders
  }

  // Hooks the controller calls to drive this view's DOM while it is mounted.
  const hooks = {
    onSend() {
      const last = ctrl.history[ctrl.history.length - 1];
      if (last && last.role === "user") bubble("user", last.content, last.shown);
      startLiveTurn();
      setComposerMode();
    },
    onStep(text) { appendStep(text); },
    onToken() { scheduleRender(); },
    onAction(rec) { renderActionCard(rec.action, rec); },
    onQuestion(q) { renderQuestionCard(q); },
    onDone(pushed) { finishRender(pushed); },
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
    titleEl.textContent = (chats.find((c) => c.id === ctrl.currentChatId) || {}).title || "New chat";
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
      ctrl.extraRuns = [];
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
    ctrl.extraRuns = [];
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
    if (!(await confirmModal({ title: "Delete chat?", danger: true, okText: "Delete",
        message: `“${c.title}” will be removed.` }))) return;
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
  ctrl.unread = false;
  changed();
  const quickRow = h("div", { class: "chat-quick" });
  const renderQuick = () => quickRow.replaceChildren(...customCommands().filter((c) => c.pinned).map((c) =>
    h("button", { class: "chat-quick-btn", title: c.description, onclick: () => { if (!ctrl.streaming) runCustom(c, ""); } }, "/" + c.name)));
  const offCommands = onCommandsChange(renderQuick);
  loadCommands().then(renderQuick);
  if (compact) window.addEventListener("hashchange", renderContext);

  // ── layout ────────────────────────────────────────────────────────────
  const newBtn = h("button", { class: "btn chat-new", onclick: () => newChat() }, icon("plus", 16), "New chat");
  const sidebar = h("aside", { class: "chat-sidebar" },
    newBtn,
    h("label", { class: "chat-project-row" }, h("span", { class: "muted" }, "New chat in"), projectSelect),
    chatList);
  const header = compact
    ? h("div", { class: "chat-header" },
        h("button", { class: "vwin-btn", title: "New chat", "aria-label": "New chat", onclick: () => newChat() }, icon("plus", 15)),
        autonomySeg, h("span", { class: "vwin-spacer" }), ...headerExtra)
    : h("div", { class: "chat-header" },
        h("div", { class: "chat-title" }, titleEl),
        h("div", { class: "chat-header-tools" }, autonomySeg,
          h("button", { class: "btn secondary small", title: "Keep the chat on top of the other tabs (also /float)",
            onclick: () => setFloating(true) }, icon("float", 14), "Float")));
  const tool = (name, title, fn, label) => h("button", { class: "chat-tool", title, "aria-label": title,
    onmousedown: (e) => e.preventDefault(), onclick: fn }, icon(name, 16), label || null);
  const composer = h("div", { class: "chat-composer" }, menu.el, attachRow, input,
    h("div", { class: "chat-tools" },
      tool("paperclip", "Attach a run so the assistant can summarize its metrics, landscape and volume motions", openRunPicker),
      tool("slash", "Commands", () => { input.value = "/"; input.focus(); menu.update("/"); }, "Commands"),
      groundBtn, h("span", { class: "vwin-spacer" }), sendBtn));
  const main = h("div", { class: "chat-main" }, header, compact ? contextRow : statusLine, log, quickRow, composer);
  const el = h("div", { class: "view chat-view" + (compact ? " compact" : "") }, compact ? null : sidebar, main);

  // Detach on route change: stop touching the DOM, but keep the socket + the
  // turn running in the background so the answer is intact when we return.
  return { el, header, cleanup: () => {
    if (ctrl.hooks === hooks) ctrl.hooks = null;
    offCommands();
    window.removeEventListener("hashchange", renderContext);
  } };
}
