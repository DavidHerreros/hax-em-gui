// AI assistant chat panel: a chat-history sidebar (new / rename / delete, grouped
// by project) beside a streaming conversation with a live activity trace, Markdown
// rendering, and a Stop button. Only routed when /api/assistant/status reports the
// optional hax-em-assistant package is installed (see app.js).
//
// Each chat's conversation + WebSocket + streaming buffers live in a module-level
// record that OUTLIVES the route view, so a question keeps running in the
// background when you leave the page or open another chat: on return the view
// rebuilds from the record and you see the ongoing answering process — or the
// finished answer.
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
import { onJobEvent } from "../events.js";

const GREETING =
  "Hi! I can help you pick programs and parameters, explain options from the docs, "
  + "diagnose a failed run's log, and reason about results. Ask me anything about your hax workflows.";

// ── persistent controller (survives view unmount) ───────────────────────────
const ctrl = {
  newChatProject: "",      // project_id for the next new chat
  grounded: true,
  hooks: null,             // set by the mounted view; null while unmounted
};
// One record per chat with live state in this tab; chats answer side by side, each on its own socket.
function blankChat(project) {
  return {
    id: null,                // persisted chat id (null = unsaved new chat)
    project: project || "",  // project_id of the chat (scopes the run picker, and where its workflows go)
    history: [],             // {role, content}
    attachedRun: null,       // {job_id,label,has_*} attached-run context (primary: full artifacts)
    extraRuns: [],           // up to 2 more runs: measured facts + parameter diff only
    // live turn state (an in-flight answer)
    streaming: false,
    turn: 0,                 // client turn id (echoed on the server's done)
    draining: null,          // a stopped turn whose late frames (up to its done) are dropped
    liveRaw: "",             // partial assistant answer accumulated so far
    liveSteps: [],           // activity-trace step texts for the current turn
    liveActions: [],         // proposed-action records for this turn
    liveQuestion: null,      // a question the assistant is waiting on (ask-and-wait)
    stopped: false,
    liveError: null,         // transient transport/model error — shown, never persisted
    turnNoAct: false,        // this turn came from a command that may not act: its proposals stay drafts
    unread: false,           // an answer finished while the chat was not on screen
    socket: null,            // opened on the first send, closed when the record is dropped
    saving: Promise.resolve(),   // this chat's persist chain
    deleted: false,
  };
}
let active = blankChat("");      // the chat on screen
const kept = new Set([active]);  // active, plus the chats answering or unread in the background
const recordOf = (id) => [...kept].find((c) => c.id === id) || null;
const view = (c) => (c === active ? ctrl.hooks : null);   // the mounted view, while it shows `c`
function drop(c) {
  kept.delete(c);
  if (c.socket) { c.socket.close(); c.socket = null; }
}
// What the floating chat's pill and the chat list show.
const watchers = new Set();
function changed() { for (const fn of watchers) fn(); }
export function onAssistantActivity(fn) { listen(); watchers.add(fn); return () => watchers.delete(fn); }
export function assistantActivity() {
  const all = [...kept];
  return { streaming: all.filter((c) => c.streaming).length, unread: all.some((c) => c.unread) || serverUnread.size > 0,
           pending: all.reduce((n, c) => n + (c.liveQuestion ? 1 : 0) + c.liveActions.filter((r) => !r.resolved).length, 0) };
}
function resolved(rec) { if (rec && !rec.resolved) { rec.resolved = true; changed(); } }

// Messages the server posts into a chat (the run monitor's findings, the assistant's reviews of them),
// also while no browser shows it: merged into its record at their place; unread unless on screen.
const serverUnread = new Set();   // chats with unseen posted messages and no record here
let listening = false;
function listen() {
  if (listening) return;
  listening = true;
  onJobEvent((m) => { if (m.kind === "chat" && m.chat_id) refreshPosted(m.chat_id); });
  api.chats().then((list) => {
    for (const c of list) if (c.unread && !recordOf(c.id)) serverUnread.add(c.id);
    changed();
  }).catch(() => {});
}
async function refreshPosted(cid) {
  const c = recordOf(cid);
  if (!c) { serverUnread.add(cid); changed(); return; }
  let got;
  try { got = await api.chat(cid); } catch { return; }
  mergePosted(c.history, got.messages || []);
  const v = view(c);
  if (v) { markSeen(c); v.onPosted(); } else c.unread = true;
  changed();
}
// As the server's chats.py merge: a posted message replaces its copy, or goes after its `after`-th own one.
function mergePosted(history, server) {
  for (const m of server) {
    if (!m.posted) continue;
    const i = history.findIndex((x) => x.posted && x.id === m.id);
    if (i >= 0) { history[i] = { ...m, seen: Boolean(history[i].seen || m.seen) }; continue; }
    let own = 0, at = history.length;
    for (let k = 0; k < history.length; k++) {
      if (own === m.after) { at = k; break; }
      if (!history[k].posted) own++;
    }
    while (at < history.length && history[at].posted) at++;
    history.splice(at, 0, { ...m });
  }
}
function markSeen(c) {
  let any = false;
  for (const m of c.history) if (m.posted && !m.seen) { m.seen = true; any = true; }
  serverUnread.delete(c.id);
  if (any) persist(c);
}

const TABS = [["/jobs", "Project"], ["/project", "Project"], ["/metrics", "Metrics"], ["/viewer", "3D viewer"],
              ["/workflows", "Workflows"], ["/compare", "Compare"], ["/settings", "Settings"], ["/assistant", ""]];
// The tab the user is on, and the run it shows when the route names one.
function describeRoute() {
  const path = currentPath();
  const hit = TABS.find(([p]) => path.startsWith(p));
  const m = /^\/jobs\/([^/]+)/.exec(path);
  return { tab: hit ? hit[1] : "Home", jobId: m ? decodeURIComponent(m[1]) : null };
}
// Workflow-card state (preview, creation) keyed by its action object, which a chat's liveActions
// keeps across re-renders: a re-rendered card shows the same state and never creates twice.
const wfCards = new WeakMap();
const stopCards = new WeakMap();   // the same for a stop card: a re-render never stops twice

function ensureSocket(c) {
  if (c.socket) return c.socket;
  // Frames of a stopped turn still on the wire (up to its done) must not land in the next one.
  const live = () => c.streaming && c.draining == null;
  c.socket = assistantChat({
    onStatus: (text) => {
      if (!live()) return;
      c.liveSteps.push(text);
      const v = view(c);
      if (v) v.onStep(text);
    },
    onAction: (action) => {
      if (!live()) return;
      const rec = { action, noAct: c.turnNoAct };
      c.liveActions.push(rec);
      changed();
      const v = view(c);
      if (v) v.onAction(rec);
    },
    onReport: (r) => {
      if (live()) api.assistantRunReport(r.job_id, true, "GET").catch((e) => toastError(e, "Report download failed."));
    },
    onQuestion: (q) => {
      if (!live()) return;
      c.liveQuestion = q;
      changed();
      const v = view(c);
      if (v) v.onQuestion(q);
    },
    onToken: (t) => {
      if (!live()) return;
      c.liveRaw += t;
      const v = view(c);
      if (v) v.onToken();
    },
    // Ignore a done that belongs to an earlier (stopped) turn: after
    // stop-then-ask, the cancelled turn's done can land while the new
    // question is already streaming and must not complete it.
    onDone: (turn) => {
      if (c.draining != null && turn === c.draining) c.draining = null;
      else if (c.streaming && (turn == null || turn === c.turn)) completeTurn(c);
    },
    onError: (err) => {
      if (!live()) return;
      // Shown in the transcript but NEVER pushed into history: a transport
      // error used to become a permanent assistant turn, re-sent to the model
      // as context on every later message in the chat.
      c.liveError = String(err);
      completeTurn(c);
    },
    onClose: () => {
      // The socket dropped (server restart, network). The turn is lost
      // server-side; unlock the composer instead of staying on "Stop" forever.
      c.draining = null;
      if (!c.streaming) return;
      c.liveError = "connection lost — the server went away mid-answer";
      completeTurn(c);
    },
  });
  return c.socket;
}

// `shown` is what the bubble displays when the text is a command's expanded prompt.
// Sent mid-turn, it interrupts that turn (kept, marked stopped) and the server starts this one at once.
function sendMessage(c, text, { shown, noAct } = {}) {
  stopTurn(c);
  c.history.push(shown ? { role: "user", content: text, shown } : { role: "user", content: text });
  c.turnNoAct = Boolean(noAct);
  c.streaming = true;
  c.turn = (c.turn || 0) + 1;
  c.liveRaw = "";
  c.liveSteps = [];
  c.liveActions = [];
  c.stopped = false;
  c.liveError = null;
  const payload = { messages: c.history.map(({ role, content }) => ({ role, content })),
                    grounded: ctrl.grounded, turn: c.turn, project_id: c.project || null };
  const here = describeRoute();
  if (here.jobId) payload.route_job_id = here.jobId;   // "stop the run" on a job's page means that job
  if (c.attachedRun) payload.run_job_id = c.attachedRun.job_id;
  if (c.extraRuns.length) payload.extra_run_job_ids = c.extraRuns.map((r) => r.job_id);
  const v = view(c);
  if (v) v.onSend();
  changed();
  // Saved at once, so a chat answering in the background is in the list; a new chat is sent once it has
  // its id, to which the server links what the turn launches or watches.
  const saved = persist(c);
  const send = () => ensureSocket(c).send({ ...payload, chat_id: c.id });
  if (c.id) send(); else saved.then(send);
}

function completeTurn(c) {
  const pushed = Boolean(c.liveRaw);
  if (pushed) c.history.push({ role: "assistant", content: c.liveRaw });
  else if (!c.stopped && c.history.length &&
           c.history[c.history.length - 1].role === "user") c.history.pop();
  c.streaming = false;
  c.liveRaw = "";
  c.liveQuestion = null;
  const v = view(c);
  if ((pushed || c.liveError) && !v) c.unread = true;
  changed();
  // `pushed` tells the view whether the live bubble has an answer to show:
  // rendering "the last assistant message" unconditionally used to duplicate
  // the PREVIOUS answer into the new bubble on an empty turn.
  if (v) v.onDone(pushed);
  persist(c);
}

function stopTurn(c) {
  if (!c.streaming) return;
  c.stopped = true;
  c.draining = c.turn;
  if (c.socket) c.socket.stop();
  c.liveSteps.push("⏹ stopped");
  const v = view(c);
  if (v) v.onStep("⏹ stopped");
  completeTurn(c);
}

function deriveTitle(c) {
  const u = c.history.find((m) => m.role === "user");
  return u ? (u.shown || u.content).slice(0, 60) : "New chat";
}
// Serialised per chat: two overlapping persists used to both see no id and
// both POST a new chat (one duplicate, one empty). Each chat writes its own
// history, whichever chat is on screen.
function persist(c) {
  c.saving = c.saving.then(async () => {
    if (!c.history.length || c.deleted) return;
    if (!c.id) c.id = (await api.createChat(c.project || null, deriveTitle(c))).id;
    await api.updateChat(c.id, { messages: c.history });
    if (ctrl.hooks && ctrl.hooks.onChatsChanged) ctrl.hooks.onChatsChanged();
  }).catch(() => { /* persistence is best-effort */ });
  return c.saving;
}

// ── the route view (rebuilds from the chat on screen; detaches on unmount) ───
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
  // Send works during a turn too (it interrupts the turn); Stop ends the turn without sending.
  const sendBtn = h("button", { class: "chat-send", title: "Send", "aria-label": "Send", onclick: () => send() },
    icon("arrowUp", 17));
  const stopBtn = h("button", { class: "chat-send danger", title: "Stop", "aria-label": "Stop",
    onclick: () => stopTurn(active) }, icon("stop", 14));
  function setComposerMode() { stopBtn.style.display = active.streaming ? "" : "none"; }

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
  const settingsLink = h("a", { href: "#/settings", class: "chat-reindex", title: "Assistant settings: provider, models, docs",
    style: { display: "inline-flex", alignItems: "center", justifyContent: "center" } }, icon("gear", 13));
  const statusLine = h("div", { class: "chat-status" }, statusText, reindexBtn, settingsLink, wsWarn);
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
    if (active.attachedRun) {
      const ar = active.attachedRun;
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
      active.extraRuns.forEach((er, i) => {
        attachRow.appendChild(h("div", { class: "chat-attach-chip extra", title: "Compared against the primary run: the model sees its measured facts and the flags that differ" },
          icon("grid", 13),
          h("span", { class: "chat-attach-name" }, er.label),
          h("button", { class: "chat-item-btn", title: "Detach", onclick: () => { active.extraRuns.splice(i, 1); renderAttachRow(); } }, icon("x", 13))));
      });
      if (active.extraRuns.length < 2) {
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
    const attached = [active.attachedRun, ...active.extraRuns].some((r) => r && r.job_id === here.jobId);
    clear(contextRow);
    append(contextRow, [icon("eye", 13),
      h("span", null, "Looking at ", h("strong", null, here.tab + (job ? ` · ${job.name || job.program}` : ""))),
      here.jobId && !attached ? h("button", { class: "chat-link", onclick: () => attachJob(here.jobId) }, "Attach this run") : null]);
  }
  async function attachJob(id) {
    if (!id) { active.attachedRun = null; active.extraRuns = []; renderAttachRow(); return; }
    let preview;
    try { preview = await api.assistantRunPreview(id); }
    catch { preview = { job_id: id, label: "run", has_metrics: false, has_latents: false, has_volumes: false }; }
    if (active.attachedRun && active.attachedRun.job_id !== id) {
      // A second/third pick becomes a comparison run rather than replacing the primary.
      if (!active.extraRuns.some((r) => r.job_id === id) && active.extraRuns.length < 2) active.extraRuns.push(preview);
    } else {
      active.attachedRun = preview;
    }
    renderAttachRow();
  }
  // Detaching the primary promotes the first comparison run (its artifacts load next turn).
  function detachRun() { active.attachedRun = active.extraRuns.shift() || null; renderAttachRow(); }
  async function downloadReport() {
    if (!active.attachedRun || reportBusy) return;
    reportBusy = true; renderAttachRow();
    try {
      await api.assistantRunReport(active.attachedRun.job_id, true);
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
      if (!active.id) { active.project = ctrl.newChatProject; renderAttachRow(); }
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
  // Rebuild the whole log from the chat on screen: history, the in-flight turn (if
  // streaming), and any proposed-action cards. Called on mount and on chat switch.
  function renderConversation() {
    log.replaceChildren();
    current = currentActivity = questionCard = null;
    if (!active.history.length && !active.streaming) { bubble("assistant", GREETING); }
    else for (const m of active.history) {
      bubble(m.role, m.content, m.shown);
      if (m.watch) findingCard(m);
      if (m.next && !m.next.used) nextStepCard(m);
    }
    if (active.liveError && !active.streaming) {   // a turn that failed while the chat was not on screen
      log.appendChild(h("div", { class: "chat-msg assistant" }, h("div", { class: "chat-error muted" }, `⚠ ${active.liveError}`)));
      active.liveError = null;
    }
    if (active.streaming) {
      startLiveTurn();
      for (const s of active.liveSteps) appendStep(s);
      if (active.liveRaw) current.replaceChildren(renderMarkdown(active.liveRaw));
    }
    for (const rec of active.liveActions) renderActionCard(rec.action, rec);
    if (active.streaming && active.liveQuestion) renderQuestionCard(active.liveQuestion);
    log.scrollTop = log.scrollHeight;
    setComposerMode();
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }
  // "/" menu: Enter runs the command, Tab (or a command that takes a value) fills it in.
  const editCommands = (select) => openCommandEditor({
    select, lastMessage: (active.history.filter((m) => m.role === "user" && !m.shown).pop() || {}).content || "",
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
      if (current) current.replaceChildren(renderMarkdown(active.liveRaw));
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
      const content = pushed && active.history.length &&
        active.history[active.history.length - 1].role === "assistant"
        ? active.history[active.history.length - 1].content : "";
      current.replaceChildren(renderMarkdown(content));
      if (active.liveError) {
        current.appendChild(h("div", { class: "chat-error muted" }, `⚠ ${active.liveError}`));
        active.liveError = null;
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
  let questionCard = null, answerPending = null;   // the composer answers a pending question too
  function dismissQuestion() {
    if (questionCard) { questionCard.remove(); questionCard = null; }
    answerPending = null;
  }
  function renderQuestionCard(q) {
    dismissQuestion();
    const chat = active;
    const controls = h("div", { class: "chat-action-actions" });
    const reply = (payload, shown) => {
      chat.liveQuestion = null;
      changed();
      ensureSocket(chat).answer({ id: q.id, ...payload });
      controls.replaceChildren(h("span", { class: "muted chat-action-note" }, shown));
      answerPending = null;
    };
    answerPending = (text) => reply({ value: text }, `Answered: ${text}`);
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

  // ── a run-monitor finding, or the assistant's review of it, posted by the server ──
  // One decision, wherever Apply is clicked (here or in the notification): the server checks it again.
  function findingCard(m) {
    const w = m.watch, chat = active;
    const act = (label, primary, call) => h("button", { class: "btn small" + (primary ? "" : " secondary"),
      onclick: async (e) => {
        e.currentTarget.disabled = true;
        try { await call(); } catch (err) { toastError(err, "Could not apply it."); }
        refreshPosted(chat.id);
      } }, label);
    const same = w.rule && w.proposed && w.proposed.restart === w.rule.restart
      && JSON.stringify(w.proposed.changes) === JSON.stringify(w.rule.changes);
    const buttons = w.resolved ? [] : [
      m.posted === "review" && w.proposed
        ? act(same ? "Apply" : "Apply the assistant's change", true, () => api.assistantApplyReview(w.job_id, w.kind)) : null,
      (m.posted === "watch" && w.needs_confirmation) || (m.posted === "review" && w.proposed && w.rule && !same)
        ? act(m.posted === "review" ? "Apply the rule's change" : "Apply", m.posted === "watch",
              () => api.assistantWatchApply(w.job_id, w.kind)) : null].filter(Boolean);
    if (!w.resolved && !buttons.length) return;
    log.appendChild(h("div", { class: "chat-msg assistant" }, h("div", { class: "chat-action-actions" },
      w.resolved ? h("div", { class: "chat-action-done" }, icon("check", 14), ` Decided: ${w.resolved.message || "applied"}`)
                 : buttons)));
  }

  // ── the next step of a run the assistant launched: Continue sends its note as the user's message ──
  function nextStepCard(m) {
    const chat = active;
    log.appendChild(h("div", { class: "chat-msg assistant" }, h("div", { class: "chat-action-actions" },
      h("button", { class: "btn small", onclick: async (e) => {
        e.currentTarget.disabled = true;
        try { await api.assistantContinueUsed(chat.id, m.id); } catch (err) { toastError(err, "Could not continue."); return; }
        m.next.used = true;
        sendMessage(chat, m.next.prompt, { shown: m.next.shown });
      } }, icon("play", 14), " Continue"))));
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
  // The project the server placed a proposal in: one the request names, else the chat's.
  const placedIn = (action) => ("project_id" in action ? action.project_id : active.project) || "";
  function renderActionCard(action, rec) {
    if (!action) return;
    if (action.kind === "prepare_motion") { renderPrepareMotionCard(action, rec); return; }
    if (action.kind === "workflow") { renderWorkflowCard(action, rec); return; }
    if (action.kind === "stop_job") { renderStopCard(action, rec); return; }
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
    if (spec.recipe) {
      body.appendChild(h("div", { class: "chat-action-row" }, h("span", { class: "muted" }, "Settings"),
        h("span", null, `from ${spec.recipe}`)));
    }
    if (spec.missing_required && spec.missing_required.length) {
      body.appendChild(h("div", { class: "chat-action-row" },
        h("span", { class: "muted" }, "You'll set"),
        h("span", { class: "chat-action-missing" },
          spec.missing_required.map((m) => "--" + m).join(", "))));
    }
    const pid = placedIn(action);
    const projName = projectName(pid);
    body.appendChild(h("div", { class: "chat-action-row" },
      h("span", { class: "muted" }, "Project"),
      h("span", null, pid ? projName : "none — pick one in the form")));
    const open = h("button", { class: "btn chat-action-open" }, icon("arrowRight", 15), "Open in job builder");
    open.addEventListener("click", () => {
      store.prefillProgram({
        program: action.program, values: spec.args || {},
        projectId: pid || null, projectName: projName,
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
      st = { request: (action.form && action.form.raw) || "", projectId: placedIn(action) || null, chatId: active.id,
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
                                                fingerprint: p.fingerprint, chatId: st.chatId })
      .then((res) => {
        st.created = { ...res, auto };
        if (!run && ctrl.hooks === hooks) openInBuilder(res.workflow_id);   // only while this view is up
      })
      .catch((e) => { st.error = wfErrText(e); })
      .finally(() => { st.creating = null; });
    renderWorkflowActions(el, st, p);
  }

  // ── stop a job ("stop the run"): the server named the job(s) and what autonomy allows ──
  // Advisory points to the job's own Stop; ask stops on the click; auto stops alone only the
  // assistant's own job. Stopping keeps the run folder and its checkpoint.
  const KEPT = "Its run folder and checkpoint are kept; nothing is deleted.";
  function renderStopCard(action, rec) {
    let st = stopCards.get(action);
    if (!st) {
      st = { pick: action.jobs.length === 1 ? action.jobs[0] : null, stopping: null, done: null, error: null,
             autoTried: false };
      stopCards.set(action, st);
    }
    const body = h("div", { class: "chat-action-body" });
    const card = h("div", { class: "chat-action" }, cardHead("stop", "muted", "", "Stop a run"), body);
    log.appendChild(h("div", { class: "chat-msg assistant" }, card));
    fillStopCard(card, body, action, st, rec);
    log.scrollTop = log.scrollHeight;
  }
  function fillStopCard(card, body, action, st, rec) {
    const j = st.pick;
    const redraw = () => fillStopCard(card, body, action, st, rec);
    const title = card.querySelector(".chat-action-title");
    clear(title);
    append(title, ["Stop ", h("strong", null, j ? j.name : "a run")]);
    clear(body);
    const line = (...kids) => body.appendChild(h("div", null, ...kids));
    const actions = (...kids) => body.appendChild(h("div", { class: "chat-action-actions" }, ...kids));
    const state = () => {
      if (!j && action.unknown) {
        line(h("code", { class: "md-code" }, action.unknown), " is not the folder of a job of the GUI, so nothing is stopped from here.");
        return ["muted", "Nothing to stop"];
      }
      if (!j && !action.jobs.length) {
        const where = action.named || (active.project && projectName(active.project));
        line("Nothing is waiting or running", where ? ` in “${where}”` : "", ", so there is nothing to stop.");
        return ["muted", "Nothing to stop"];
      }
      if (!j) {
        line("Several jobs are waiting or running. Which one should stop?");
        actions(...action.jobs.map((x) => h("button", { class: "btn small secondary",
          onclick: () => { st.pick = x; redraw(); } }, `${x.name} (${x.status})`)));
        return ["ask", "Pick one"];
      }
      const openJob = h("button", { class: "chat-link", onclick: () => navigate(`/jobs/${encodeURIComponent(j.id)}`) }, "Open the job");
      if (st.done) {
        body.appendChild(h("div", { class: "chat-action-done" }, icon("check", 14),
          `${st.done.auto ? "Autonomy is auto and the assistant launched this job, so it stopped it" : "Stopped"}: `
          + `“${j.name}” is now ${st.done.status}. ${KEPT}`));
        return st.done.auto ? ["done", "Stopped by the assistant"] : ["done", "Stopped"];
      }
      if (!j.decision) {
        line(`“${j.name}” already ended (${j.status}): there is nothing to stop.`);
        return ["muted", "Ended"];
      }
      if (j.decision === "propose") {
        line(`The assistant does not stop it: ${j.reason}. Use the Stop button on the job's page. ${KEPT}`);
        actions(openJob);
        return ["proposal", "Proposal"];
      }
      if (st.stopping) {
        line(h("span", { class: "muted" }, "Stopping…"));
        st.stopping.then(redraw);
        return ["ask", "Stopping"];
      }
      line(h("span", { class: "muted" }, `${j.program} · ${j.status}. ${KEPT}`));
      actions(h("button", { class: "btn danger chat-action-open", onclick: () => stop(false) }, icon("stop", 14), `Stop “${j.name}”`),
        openJob, st.error ? h("div", { class: "chat-action-warn" }, st.error) : null);
      return ["ask", "Needs your OK"];
    };
    // The click is the confirmation; at auto the card acts once, on the assistant's own job only.
    const stop = (auto) => {
      st.error = null;
      st.stopping = api.assistantStopJob(j.id, auto).then((r) => { st.done = r; }, (e) => { st.error = errText(e); })
        .finally(() => { st.stopping = null; });
      redraw();
    };
    if (j && j.decision === "proceed" && !st.done && !st.autoTried) { st.autoTried = true; stop(true); return; }
    const [kind, text] = state();
    card.querySelector(".chat-badge").replaceWith(badge(kind, text));
    if (kind !== "ask") resolved(rec);
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
    if (cmd.attach_run && !active.attachedRun && here.jobId) await attachJob(here.jobId);
    const text = expandCommand(cmd, {
      run: active.attachedRun && active.attachedRun.label, tab: here.tab, text: arg,
      project: active.project ? projectName(active.project) : "",
    });
    sendMessage(active, text, { shown: `/${cmd.name}${arg ? " " + arg : ""}`, noAct: !cmd.may_act });
  }
  // True when `name` was a command and has been handled: the model never sees the line.
  function runCommand(name, arg) {
    if (name === "autonomy") autonomyCommand(arg);
    else if (name === "attach") openRunPicker();
    else if (name === "report") { if (active.attachedRun) downloadReport(); else notice("report: attach a run first (/attach)"); }
    else if (name === "float") { setFloating(!compact); if (compact) navigate("/assistant"); }
    else if (name === "new") newChat();
    else if (name === "commands") editCommands();
    else {
      const cmd = findCommand(name);
      if (!cmd) return false;
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
    if (!text) return;
    input.value = "";
    autosize();
    if (answerPending && active.liveQuestion) { answerPending(text); return; }   // an answer, not a new request
    sendMessage(active, text);   // the chat's history + socket turn; onSend renders
  }

  // Hooks the controller calls to drive this view's DOM while it is mounted.
  const hooks = {
    onSend() {
      const last = active.history[active.history.length - 1];
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
    onPosted() { renderConversation(); },
  };

  // ── persistence sidebar ───────────────────────────────────────────────
  function projectName(pid) {
    if (!pid) return "No project";
    const p = projects.find((x) => x.id === pid);
    return p ? p.name : "Unknown project";
  }
  async function refreshChats() {
    try { chats = await api.chats(); } catch { chats = []; }
    for (const c of chats) if (c.unread && c.id !== active.id && !recordOf(c.id)) serverUnread.add(c.id);
    renderChatList();
  }
  function renderChatList() {
    titleEl.textContent = (chats.find((c) => c.id === active.id) || {}).title || "New chat";
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
    // Answering (pulsing, as a running job), waiting on your answer, or finished while not on screen (dot).
    const rec = recordOf(c.id);
    const state = !rec ? null
      : rec.liveQuestion ? h("span", { class: "chat-badge ask" }, "needs answer")
      : rec.streaming ? h("i", { class: "jt-dot running", title: "Answering…" })
      : rec.unread ? h("span", { class: "chat-step-dot", title: "New reply" }) : null;
    const posted = !rec && serverUnread.has(c.id) ? h("span", { class: "chat-step-dot", title: "New finding" }) : null;
    const renameBtn = h("button", { class: "chat-item-btn", title: "Rename",
      onclick: (e) => { e.stopPropagation(); renameChat(c); } }, icon("pencil", 14));
    const delBtn = h("button", { class: "chat-item-btn", title: "Delete",
      onclick: (e) => { e.stopPropagation(); removeChat(c); } }, icon("trash", 14));
    return h("div", {
      class: "chat-item" + (c.id === active.id ? " active" : ""),
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
      h("span", { class: "chat-item-title", title: c.title }, c.title), state || posted,
      h("span", { class: "chat-item-actions" }, renameBtn, delBtn));
  }
  async function moveChatToProject(chatId, projectId) {
    const c = chats.find((x) => x.id === chatId);
    if (c && (c.project_id || "") === (projectId || "")) return;  // already there
    try {
      await api.updateChat(chatId, { project_id: projectId || null });
      const rec = recordOf(chatId);
      if (rec) rec.project = projectId || "";
      if (rec === active) renderAttachRow();
      await refreshChats();
    } catch { /* ignore */ }
  }
  // Another chat on screen; the one left keeps answering in the background, or is dropped when idle
  // (a card still waiting for you keeps it, so its button is there when you come back).
  function show(c) {
    const left = active;
    active = c;
    kept.add(c);
    c.unread = false;
    markSeen(c);
    if (left !== c && !left.streaming && !left.unread && left.liveActions.every((r) => r.resolved)) drop(left);
    changed();
    renderConversation();
    renderChatList();
    renderAttachRow();
  }
  async function selectChat(id) {
    if (id === active.id) return;
    let c = recordOf(id);
    if (!c) {
      try {
        const got = await api.chat(id);
        c = Object.assign(blankChat(got.project_id), { id: got.id,
          history: Array.isArray(got.messages) ? got.messages.slice() : [] });
      } catch { return; }
    }
    show(c);
  }
  function newChat() {
    show(blankChat(ctrl.newChatProject));
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
    const rec = recordOf(c.id);
    if (rec) { rec.deleted = true; stopTurn(rec); }   // a running chat stops first; nothing is saved again
    try {
      await api.deleteChat(c.id);
      if (rec === active) show(blankChat(ctrl.newChatProject));
      else if (rec) { drop(rec); changed(); }
      await refreshChats();
    } catch { if (rec) rec.deleted = false; }
  }

  function buildProjectSelect() {
    projectSelect.replaceChildren(
      h("option", { value: "" }, "No project"),
      ...projects.map((p) => h("option", { value: p.id, selected: p.id === ctrl.newChatProject }, p.name)));
  }

  // ── init (mount): rebuild from the chat on screen, then attach hooks ──
  api.projects().then((p) => { projects = p || []; buildProjectSelect(); renderChatList(); }).catch(() => {});
  loadJobs().then(renderAttachRow);
  refreshChats();
  renderConversation();
  renderAttachRow();
  ctrl.hooks = hooks;   // start receiving live updates for this mount
  listen();
  active.unread = false;
  changed();
  const quickRow = h("div", { class: "chat-quick" });
  const renderQuick = () => quickRow.replaceChildren(...customCommands().filter((c) => c.pinned).map((c) =>
    h("button", { class: "chat-quick-btn", title: c.description, onclick: () => runCustom(c, "") }, "/" + c.name)));
  const offCommands = onCommandsChange(renderQuick);
  const offActivity = onAssistantActivity(() => { if (!compact && !dragChatId) renderChatList(); });
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
      groundBtn, h("span", { class: "vwin-spacer" }), stopBtn, sendBtn));
  const main = h("div", { class: "chat-main" }, header, compact ? contextRow : statusLine, log, quickRow, composer);
  const el = h("div", { class: "view chat-view" + (compact ? " compact" : "") }, compact ? null : sidebar, main);

  // Detach on route change: stop touching the DOM, but keep the socket + the
  // turn running in the background so the answer is intact when we return.
  return { el, header, cleanup: () => {
    if (ctrl.hooks === hooks) ctrl.hooks = null;
    offCommands();
    offActivity();
    window.removeEventListener("hashchange", renderContext);
  } };
}
