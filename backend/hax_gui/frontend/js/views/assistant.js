// AI assistant chat panel: a chat-history sidebar (new / rename / delete, grouped
// by project) beside a streaming conversation with a live activity trace, Markdown
// rendering, and a Stop button. Only routed when /api/assistant/status reports the
// optional hax-em-assistant package is installed (see app.js).
import { h } from "../h.js";
import { icon } from "../icons.js";
import { assistantChat, api } from "../api.js";
import { renderMarkdown } from "../markdown.js";

const GREETING =
  "Hi! I can help you pick programs and parameters, explain options from the docs, "
  + "diagnose a failed run's log, and reason about results. Ask me anything about your hax workflows.";

export function AssistantView() {
  let history = [];            // {role, content} — the active conversation
  let socket = null;
  let streaming = false;
  let grounded = true;
  let currentChatId = null;    // persisted chat id (null = unsaved new chat)
  let chats = [];              // chat metadata for the sidebar
  let projects = [];           // {id, name} for grouping + the new-chat selector
  let newChatProject = "";     // project_id for the next new chat ("" = none)
  let attachedRun = null;      // {job_id,label,has_metrics,has_latents,has_volumes}
  let currentChatProject = ""; // project_id of the active chat (scopes the run picker)
  let jobsCache = [];          // jobs for the attach-run picker

  // ── elements ──────────────────────────────────────────────────────────
  const log = h("div", { class: "chat-log" });
  const input = h("textarea", {
    class: "chat-input",
    rows: "1",
    placeholder: "Ask about programs, parameters, a run, a log… (Enter to send, Shift+Enter for newline)",
  });
  const sendBtn = h("button", { class: "btn chat-send" }, "Send");
  sendBtn.addEventListener("click", () => { if (streaming) stopStreaming(); else send(); });
  function setComposerMode() {
    sendBtn.textContent = streaming ? "Stop" : "Send";
    sendBtn.classList.toggle("danger", streaming);
  }

  const groundBtn = h("button", {
    class: "chip active",
    title: "Ground answers in the hax docs + program schema",
    onclick: () => { grounded = !grounded; groundBtn.classList.toggle("active", grounded); },
  }, "docs grounding");

  const statusLine = h("div", { class: "chat-status" }, "");
  api.assistantStatus().then((s) => {
    statusLine.textContent = s.available
      ? `model: ${s.model}  ·  docs: ${s.docs_grounding ? "on" : "off"}  ·  ${s.programs} programs, ${s.doc_chunks} doc chunks`
      : "assistant unavailable";
  }).catch(() => { statusLine.textContent = ""; });

  // ── attach a run (explicit): scope to the chat's project when it has one ──
  const attachRow = h("div", { class: "chat-attach" });
  async function loadJobs() {
    try { jobsCache = await api.jobs(); } catch { jobsCache = []; }
  }
  function relevantJobs() {
    const js = currentChatProject
      ? jobsCache.filter((j) => j.project_id === currentChatProject)
      : jobsCache.slice();
    return js;  // api.jobs() is already newest-first
  }
  function jobOptionLabel(j) {
    const bits = [j.program];
    if (j.name && j.name !== j.program) bits.push(j.name);
    if (j.status && j.status !== "succeeded") bits.push(`(${j.status})`);
    return bits.join(" · ");
  }
  function renderAttachRow() {
    attachRow.replaceChildren();
    if (attachedRun) {
      const tags = ["metrics", "latents", "volumes"].filter((k) => attachedRun["has_" + k])
        .map((k) => (k === "latents" && attachedRun.latents_from) ? `latents ← ${attachedRun.latents_from}` : k);
      const chip = h("div", { class: "chat-attach-chip" },
        icon("paperclip", 14),
        h("span", { class: "chat-attach-name" }, attachedRun.label),
        h("span", { class: "chat-attach-tags" },
          tags.length ? tags.join(" · ") : "no readable artifacts"),
        h("button", { class: "chat-item-btn", title: "Detach run", onclick: detachRun }, icon("x", 14)));
      attachRow.appendChild(chip);
      return;
    }
    const opts = [h("option", { value: "" }, "— attach a run —")];
    for (const j of relevantJobs()) opts.push(h("option", { value: j.id }, jobOptionLabel(j)));
    attachRow.appendChild(h("span", { class: "muted chat-attach-lead" }, icon("paperclip", 13), "Run"));
    attachRow.appendChild(h("select", {
      class: "chat-attach-select",
      title: "Attach a run so the assistant can summarize its metrics, landscape and volume motions",
      onchange: (e) => attachJob(e.target.value),
    }, ...opts));
  }
  async function attachJob(id) {
    if (!id) { attachedRun = null; renderAttachRow(); return; }
    try { attachedRun = await api.assistantRunPreview(id); }
    catch { attachedRun = { job_id: id, label: "run", has_metrics: false, has_latents: false, has_volumes: false }; }
    renderAttachRow();
  }
  function detachRun() { attachedRun = null; renderAttachRow(); }

  const chatList = h("div", { class: "chat-list" });
  const projectSelect = h("select", {
    class: "chat-project-select",
    onchange: (e) => {
      newChatProject = e.target.value;
      // an unsaved chat follows the picker's project scope immediately
      if (!currentChatId) { currentChatProject = newChatProject; renderAttachRow(); }
    },
  });

  // ── conversation rendering ────────────────────────────────────────────
  function fillBody(body, role, text) {
    if (role === "user") body.textContent = text;
    else body.replaceChildren(renderMarkdown(text));
  }
  function bubble(role, text) {
    const body = h("div", { class: "chat-body" });
    fillBody(body, role, text);
    log.appendChild(h("div", { class: `chat-msg ${role}` },
      h("div", { class: "chat-role" }, role === "user" ? "You" : "Assistant"), body));
    log.scrollTop = log.scrollHeight;
    return body;
  }
  function renderConversation() {
    log.replaceChildren();
    if (!history.length) { bubble("assistant", GREETING); return; }
    for (const m of history) bubble(m.role, m.content);
    log.scrollTop = log.scrollHeight;
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });

  // ── streaming turn (activity trace + progressive Markdown) ────────────
  function startAssistantTurn() {
    const activity = h("div", { class: "chat-activity" });
    const body = h("div", { class: "chat-body streaming" });
    log.appendChild(h("div", { class: "chat-msg assistant" },
      h("div", { class: "chat-role" }, "Assistant"), activity, body));
    log.scrollTop = log.scrollHeight;
    return { activity, body };
  }
  function addStep(activity, text) {
    const prev = activity.lastElementChild;
    if (prev) prev.classList.remove("live");
    activity.appendChild(h("div", { class: "chat-step live" },
      h("span", { class: "chat-step-dot" }), h("span", null, text)));
    log.scrollTop = log.scrollHeight;
  }

  let current = null, currentActivity = null, currentRaw = "", raf = 0;
  function scheduleRender() {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      current.replaceChildren(renderMarkdown(currentRaw));
      log.scrollTop = log.scrollHeight;
    });
  }
  function finishRender() {
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    current.replaceChildren(renderMarkdown(currentRaw));
    current.classList.remove("streaming");
    if (currentActivity && currentActivity.lastElementChild) currentActivity.lastElementChild.classList.remove("live");
    log.scrollTop = log.scrollHeight;
  }
  function completeTurn() {
    finishRender();
    if (currentRaw) history.push({ role: "assistant", content: currentRaw });
    else if (history.length && history[history.length - 1].role === "user") history.pop();
    streaming = false;
    setComposerMode();
    persistCurrent();
  }
  function stopStreaming() {
    if (!streaming) return;
    if (socket) socket.stop();
    if (currentActivity) addStep(currentActivity, "⏹ stopped");
    completeTurn();
  }

  function ensureSocket() {
    if (socket) return socket;
    socket = assistantChat({
      onStatus: (text) => { if (streaming && currentActivity) addStep(currentActivity, text); },
      onToken: (t) => { if (!streaming) return; currentRaw += t; scheduleRender(); },
      onDone: () => { if (streaming) completeTurn(); },
      onError: (err) => {
        if (!streaming) return;
        currentRaw += (currentRaw ? "\n\n" : "") + `⚠ ${err}`;
        completeTurn();
      },
    });
    return socket;
  }

  function send() {
    const text = input.value.trim();
    if (!text || streaming) return;
    history.push({ role: "user", content: text });
    bubble("user", text);
    input.value = "";
    autosize();
    streaming = true;
    setComposerMode();
    currentRaw = "";
    const turn = startAssistantTurn();
    current = turn.body;
    currentActivity = turn.activity;
    const payload = { messages: history, grounded };
    if (attachedRun) payload.run_job_id = attachedRun.job_id;
    ensureSocket().send(payload);
  }

  // ── persistence + sidebar ─────────────────────────────────────────────
  function deriveTitle() {
    const u = history.find((m) => m.role === "user");
    return u ? u.content.slice(0, 60) : "New chat";
  }
  async function persistCurrent() {
    if (!history.length) return;
    try {
      if (!currentChatId) {
        const c = await api.createChat(newChatProject || null, deriveTitle());
        currentChatId = c.id;
      }
      await api.updateChat(currentChatId, { messages: history });
      await refreshChats();
    } catch { /* persistence is best-effort */ }
  }

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
    if (!chats.length) { chatList.appendChild(h("div", { class: "chat-empty" }, "No chats yet")); return; }
    const groups = new Map();
    for (const c of chats) {
      const k = c.project_id || "";
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(c);
    }
    const keys = [...groups.keys()].sort((a, b) => projectName(a).localeCompare(projectName(b)));
    for (const k of keys) {
      chatList.appendChild(h("div", { class: "chat-group-title" }, projectName(k)));
      for (const c of groups.get(k)) chatList.appendChild(chatItem(c));
    }
  }
  function chatItem(c) {
    const renameBtn = h("button", { class: "chat-item-btn", title: "Rename",
      onclick: (e) => { e.stopPropagation(); renameChat(c); } }, icon("pencil", 14));
    const delBtn = h("button", { class: "chat-item-btn", title: "Delete",
      onclick: (e) => { e.stopPropagation(); removeChat(c); } }, icon("trash", 14));
    return h("div", { class: "chat-item" + (c.id === currentChatId ? " active" : ""),
      onclick: () => selectChat(c.id) },
      h("span", { class: "chat-item-title", title: c.title }, c.title),
      h("span", { class: "chat-item-actions" }, renameBtn, delBtn));
  }
  async function selectChat(id) {
    if (streaming) return;
    try {
      const c = await api.chat(id);
      currentChatId = c.id;
      currentChatProject = c.project_id || "";
      attachedRun = null;
      history = Array.isArray(c.messages) ? c.messages.slice() : [];
      renderConversation();
      renderChatList();
      renderAttachRow();
    } catch { /* ignore */ }
  }
  function newChat() {
    if (streaming) return;
    currentChatId = null;
    currentChatProject = newChatProject || "";
    attachedRun = null;
    history = [];
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
      if (c.id === currentChatId) { currentChatId = null; history = []; renderConversation(); }
      await refreshChats();
    } catch { /* ignore */ }
  }

  function buildProjectSelect() {
    projectSelect.replaceChildren(
      h("option", { value: "" }, "No project"),
      ...projects.map((p) => h("option", { value: p.id }, p.name)));
  }

  // ── init ──────────────────────────────────────────────────────────────
  api.projects().then((p) => { projects = p || []; buildProjectSelect(); renderChatList(); }).catch(() => {});
  loadJobs().then(renderAttachRow);
  refreshChats();
  renderConversation();
  renderAttachRow();

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

  return { el, cleanup: () => { if (socket) socket.close(); } };
}
