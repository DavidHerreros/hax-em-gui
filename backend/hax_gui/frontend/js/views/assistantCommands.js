// The chat's "/" commands: the built-in ones, the user's own (one text file each on
// the server), the menu that opens when "/" is typed, and the editor.
import { h, clear, append } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errText } from "../toast.js";
import { confirmModal } from "./confirmModal.js";

export const BUILTINS = [
  { name: "autonomy", description: "Advisory, ask first or auto", arg: true },
  { name: "attach", description: "Attach a run to the conversation" },
  { name: "report", description: "PDF report of the attached run" },
  { name: "float", description: "Keep the chat on top of the other tabs" },
  { name: "new", description: "Start a new chat" },
  { name: "commands", description: "Create and edit your own commands" },
];
// replaceChildren would print a skipped (null) child as the text "null".
const fill = (el, ...kids) => { clear(el); append(el, kids); };
const NAME_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;
const VARS = ["{run}", "{project}", "{tab}", "{text}"];

let cache = [];
let loading = null;
const listeners = new Set();
export function customCommands() { return cache; }
export function findCommand(name) { return cache.find((c) => c.name === name) || null; }
export function onCommandsChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }
export function loadCommands(force = false) {
  if (!loading || force) {
    loading = api.assistantCommands().then((list) => { cache = list || []; }, () => { cache = []; })
      .then(() => { for (const fn of listeners) fn(); });
  }
  return loading;
}

// The prompt the model gets: the variables filled in, and whatever was typed after
// the command appended when the prompt has no {text} of its own.
export function expandCommand(cmd, { run, project, tab, text }) {
  const vals = { run: run || "(no run attached)", project: project || "(no project)", tab: tab || "", text: text || "" };
  let out = cmd.prompt.replace(/\{(run|project|tab|text)\}/g, (_, k) => vals[k]).trim();
  if (text && !cmd.prompt.includes("{text}")) out += "\n\n" + text;
  if (cmd.include_tab && tab && !cmd.prompt.includes("{tab}")) out += `\n\n(I am on the ${tab} tab.)`;
  return out;
}

// ── the menu above the message box ──────────────────────────────────────────
// update(value) opens it while the box holds only "/word"; key(e) handles the arrow
// keys, Enter (run), Tab (fill in) and Escape, and says whether it took the key.
export function slashMenu({ onPick, onNew, onManage, autonomy }) {
  const el = h("div", { class: "chat-slash", role: "listbox", hidden: true });
  let items = [], sel = 0;

  function row(item, i) {
    const custom = !BUILTINS.includes(item);
    return h("div", {
      class: "chat-slash-row" + (i === sel ? " sel" : ""), role: "option", "aria-selected": String(i === sel),
      onmousedown: (e) => { e.preventDefault(); onPick(item, false); },
      onmousemove: () => { if (sel !== i) { sel = i; render(); } },
    },
      h("code", { class: custom ? "custom" : "" }, "/" + item.name),
      h("span", { class: "chat-slash-desc" }, item.description || ""),
      custom && item.may_act ? h("span", { class: "chat-badge ask" }, "can launch jobs") : null,
      item.name === "autonomy" && autonomy() ? h("span", { class: "muted" }, "now: " + autonomy()) : null);
  }
  function render() {
    const mine = items.filter((it) => !BUILTINS.includes(it));
    const built = items.filter((it) => BUILTINS.includes(it));
    const foot = h("div", { class: "chat-slash-foot" },
      h("span", { class: "muted" }, "Arrows to move · Enter to run · Tab to fill in"),
      h("button", { class: "btn ghost small", onmousedown: (e) => { e.preventDefault(); onManage(); } }, "Manage"),
      h("button", { class: "btn secondary small", onmousedown: (e) => { e.preventDefault(); onNew(); } }, icon("plus", 13), "New command"));
    fill(el,
      mine.length ? h("div", { class: "chat-slash-title" }, "Your commands") : null,
      ...mine.map((it) => row(it, items.indexOf(it))),
      built.length ? h("div", { class: "chat-slash-title" }, "Built in") : null,
      ...built.map((it) => row(it, items.indexOf(it))),
      items.length ? null : h("div", { class: "chat-slash-title" }, "No command starts with that"),
      foot);
  }
  function close() { el.hidden = true; }
  function update(value) {
    const m = /^\/([a-z0-9-]*)$/i.exec(value);
    if (!m) { close(); return; }
    const q = m[1].toLowerCase();
    items = [...cache, ...BUILTINS].filter((it) => it.name.startsWith(q));
    sel = Math.min(sel, Math.max(items.length - 1, 0));
    el.hidden = false;
    render();
  }
  function key(e) {
    if (el.hidden) return false;
    if (e.key === "Escape") { close(); return true; }
    if (!items.length) return false;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      sel = (sel + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length;
      render();
    } else if (e.key === "Enter" && !e.shiftKey) onPick(items[sel], false);
    else if (e.key === "Tab") onPick(items[sel], true);
    else return false;
    e.preventDefault();
    return true;
  }
  return { el, update, key, close };
}

// ── the editor ──────────────────────────────────────────────────────────────
// `lastMessage` offers "save the last message as a command"; `onTry(draft)` runs the
// unsaved draft in the chat.
export function openCommandEditor({ select = null, lastMessage = "", onTry } = {}) {
  const blank = (prompt = "") => ({ orig: null, name: "", description: "", prompt,
                                    attach_run: true, include_tab: false, may_act: false, pinned: false });
  let cur = blank();
  const list = h("div", { class: "cmd-list" });
  const form = h("div", { class: "cmd-form" });
  const overlay = h("div", { class: "overlay", onclick: close },
    h("div", { class: "modal cmd-editor", role: "dialog", "aria-label": "Your commands", onclick: (e) => e.stopPropagation() },
      h("div", { class: "cmd-side" }, h("div", { class: "cmd-side-title" }, "Your commands"), list),
      form));
  function onKey(e) { if (e.key === "Escape" && document.querySelectorAll(".overlay").length === 1) close(); }
  function close() { overlay.remove(); document.removeEventListener("keydown", onKey); }

  function edit(c) { cur = c; renderList(); renderForm(); }
  function renderList() {
    fill(list,
      ...cache.map((c) => h("button", { class: "cmd-item" + (cur.orig === c.name ? " active" : ""),
        onclick: () => edit({ ...c, orig: c.name }) },
        h("code", null, "/" + c.name), h("span", { class: "muted" }, c.description || "No description"))),
      cache.length ? null : h("div", { class: "muted cmd-empty" }, "None yet. A command is a prompt you reuse, run by typing its name."),
      h("span", { class: "vwin-spacer" }),
      h("button", { class: "btn secondary small cmd-new", onclick: () => edit(blank()) }, icon("plus", 14), "New command"),
      lastMessage ? h("button", { class: "btn ghost small", onclick: () => edit(blank(lastMessage)) },
        "Save the last message as a command") : null);
  }

  function renderForm() {
    const name = h("input", { id: "cmd-name", value: cur.name, placeholder: "check-run", spellcheck: "false", autocomplete: "off" });
    const desc = h("input", { id: "cmd-desc", class: "cmd-input", value: cur.description, placeholder: "One line, shown in the menu" });
    const prompt = h("textarea", { id: "cmd-prompt", class: "cmd-input", rows: "6",
      placeholder: "Look at {run}. Tell me if the training is healthy and the single next step you recommend." });
    prompt.value = cur.prompt;
    const check = (key, label) => {
      const box = h("input", { type: "checkbox", checked: cur[key] });
      box.addEventListener("change", () => { cur[key] = box.checked; });
      return h("label", { class: "cmd-check" }, box, label);
    };
    const act = (value, label) => {
      const radio = h("input", { type: "radio", name: "cmd-act", checked: cur.may_act === value });
      radio.addEventListener("change", () => { cur.may_act = value; });
      return h("label", { class: "cmd-check" }, radio, label);
    };
    const insert = (v) => h("button", { class: "cmd-var", type: "button", onclick: () => {
      prompt.setRangeText(v, prompt.selectionStart, prompt.selectionEnd, "end");
      prompt.focus();
    } }, v);
    const error = h("span", { class: "chat-action-warn" });
    const read = () => Object.assign(cur, { name: name.value.trim().replace(/^\//, "").toLowerCase(),
                                           description: desc.value.trim(), prompt: prompt.value.trim() });
    const valid = (needName) => {
      read();
      error.textContent = needName && !NAME_RE.test(cur.name) ? "The name is 1-32 lowercase letters, digits or dashes."
        : needName && BUILTINS.some((b) => b.name === cur.name) ? `/${cur.name} is a built-in command.`
        : !cur.prompt ? "Write what the assistant should do." : "";
      return !error.textContent;
    };
    async function save() {
      if (!valid(true)) return;
      const { orig, name: n, ...body } = cur;
      try {
        if (orig !== n && findCommand(n) && !(await confirmModal({ title: `Replace /${n}?`, okText: "Replace",
            message: "A command with this name already exists." }))) return;
        await api.saveAssistantCommand(n, body);
        if (orig && orig !== n) await api.deleteAssistantCommand(orig);
        await loadCommands(true);
        edit({ ...cur, orig: n });
      } catch (e) { error.textContent = errText(e); }
    }
    async function remove() {
      if (!(await confirmModal({ title: `Delete /${cur.orig}?`, danger: true, okText: "Delete",
          message: "Its file is removed from the server." }))) return;
      try { await api.deleteAssistantCommand(cur.orig); await loadCommands(true); edit(blank()); }
      catch (e) { error.textContent = errText(e); }
    }
    fill(form,
      h("div", { class: "cmd-fields" },
        h("div", { class: "cmd-row" },
          h("div", { class: "cmd-field cmd-field-name" }, h("label", { for: "cmd-name" }, "Name"),
            h("div", { class: "cmd-name" }, h("span", { class: "muted" }, "/"), name)),
          h("div", { class: "cmd-field" }, h("label", { for: "cmd-desc" }, "Shown in the menu as"), desc)),
        h("div", { class: "cmd-field" }, h("label", { for: "cmd-prompt" }, "What the assistant should do"), prompt,
          h("div", { class: "cmd-vars muted" }, "Insert", ...VARS.map(insert),
            h("span", null, "{text} is whatever you type after the command."))),
        h("div", { class: "cmd-row" },
          h("fieldset", { class: "cmd-set" }, h("legend", null, "Send along"),
            check("attach_run", "The run I am looking at"), check("include_tab", "The tab I am on")),
          h("fieldset", { class: "cmd-set" }, h("legend", null, "May it act on jobs?"),
            act(false, "No, proposals stay drafts"), act(true, "Yes, following the autonomy setting"))),
        check("pinned", "Also show it as a quick button above the message box")),
      h("div", { class: "cmd-foot" },
        error, h("span", { class: "vwin-spacer" }),
        cur.orig ? h("button", { class: "btn ghost small cmd-delete", onclick: remove }, "Delete") : null,
        onTry ? h("button", { class: "btn secondary small", onclick: () => { if (valid(false)) { close(); onTry({ ...cur, name: cur.name || "try" }); } } }, "Try it") : null,
        h("button", { class: "btn small", onclick: save }, "Save"),
        h("button", { class: "btn-icon", style: { border: "none" }, title: "Close", "aria-label": "Close", onclick: close }, icon("x", 18))));
  }

  document.body.appendChild(overlay);
  document.addEventListener("keydown", onKey);
  renderList();
  renderForm();
  loadCommands().then(() => {
    const c = select && findCommand(select);
    if (c) edit({ ...c, orig: c.name }); else renderList();
  });
}
