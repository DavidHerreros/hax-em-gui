// Settings → Assistant: provider, endpoint, API key (write-only), chat and embedding models, docs.
// The key goes to the server once and is never read back; model lists are fetched by the server.
import { h, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errorBox } from "./ui.js";
import { openFilePicker } from "./filePicker.js";

const FIELDS = ["backend", "base_url", "model", "embed_model", "embed_base_url", "docs"];
const PROVIDERS = [["ollama", "Ollama"], ["openai", "OpenAI-compatible"]];
const URL_HELP = {
  ollama: "An Ollama server. On this machine the GUI starts it when it isn't running.",
  openai: "Any server that speaks the OpenAI API: OpenAI, vLLM, LM Studio or a gateway. Anthropic's own API is not supported yet.",
};

function originOf(url) {
  try {
    const u = new URL((url || "").trim());
    return /^https?:$/.test(u.protocol) && u.host ? u.origin : null;
  } catch { return null; }
}
const isLocalHost = (host) => /^(localhost|127\.[\d.]+|\[::1\])(:\d+)?$/i.test(host);

export function AssistantSettings() {
  const root = h("div", { id: "setting-assistant" });
  let cfg = null;          // server view: values, sources, key origins, docs; never a key
  let v = {};              // the form
  let lists = null;        // {chat: {...}, embed: {...}} from /api/assistant/models
  let loading = false;
  let other = { model: false, embed_model: false };
  let showHidden = false;
  let keyEdit = false, confirmRemove = false;
  let message = null, docsMsg = null;
  const keySlot = h("div");

  api.assistantConfig().then((c) => { take(c); render(); refresh(); }).catch(() => clear(root));
  return root;

  function take(c) {
    cfg = c;
    v = { ...c.values };
    keyEdit = false; confirmRemove = false;
  }
  function locked(k) { return cfg.sources[k] === "env"; }
  function docsMode() { return ["auto", "online", "off"].includes(v.docs) ? v.docs : "local"; }

  function render() {
    const focusId = document.activeElement && document.activeElement.id;
    clear(root);
    root.append(h("div", { class: "card", style: { padding: "22px", margin: "28px 0 16px" } },
      h("h2", { class: "section" }, "Assistant"),
      h("p", { class: "field-help", style: { margin: "0 0 16px" } },
        "Which model the assistant uses and where it reads the docs. Applies to new messages; open chats keep their history."),
      h("div", { class: "asst-fields" },
        providerField(), endpointField(), keyField(), modelField("model"), embedSourceField(), modelField("embed_model"), docsField()),
      h("div", { class: "row", style: { marginTop: "20px", gap: "10px", flexWrap: "wrap" } },
        h("button", { class: "btn", onclick: save }, icon("check", 18), h("span", null, "Save")),
        h("button", { class: "btn secondary", onclick: refresh, disabled: loading }, icon("refresh", 15),
          h("span", null, loading ? "Listing…" : "Test connection"))),
      message ? h("div", { style: { marginTop: "12px" } }, message) : null));
    if (focusId) document.getElementById(focusId)?.focus({ preventScroll: true });
  }

  function label(name, key) {
    return h("div", { class: "field-label" }, h("span", { class: "flag" }, name),
      h("span", { class: "ctype" }, cfg.env_vars[key] || "HAX_ASSISTANT_API_KEY"));
  }
  function lockBox(key, shown) {
    return h("div", { class: "asst-lock" }, h("div", { class: "val" }, shown || "(empty)"),
      h("div", { class: "why" }, `Set by ${cfg.env_vars[key]} in the server environment. Unset it and restart the GUI to choose here.`));
  }
  function status(kind, text) { return h("div", { class: `asst-status ${kind}` }, text); }

  function providerField() {
    if (locked("backend")) return h("div", null, label("provider", "backend"), lockBox("backend", v.backend));
    return h("div", null, label("provider", "backend"),
      h("div", { class: "chat-seg", role: "group", "aria-label": "Provider", style: { display: "inline-flex" } },
        ...PROVIDERS.map(([k, name]) => h("button", {
          class: v.backend === k ? "active" : null, "aria-pressed": String(v.backend === k),
          onclick: () => {
            if (v.backend === k) return;
            v.backend = k;
            if (!locked("base_url")) v.base_url = k === "ollama" ? cfg.defaults.base_url.ollama : "";   // a hosted endpoint is always typed, never assumed
            if (!locked("model")) v.model = "";
            if (!locked("embed_model") && !v.embed_base_url) v.embed_model = "";
            lists = null; other = { model: false, embed_model: false }; keyEdit = false; confirmRemove = false;
            render(); refresh();
          },
        }, name))));
  }

  function endpointField() {
    const chat = lists && lists.chat;
    const reach = chat ? (chat.ok
      ? status("ok", `Reachable${chat.server ? ` · ${chat.server}` : ""} · ${chat.chat.length + chat.embed.length} models listed`)
      : status("err", chat.error)) : null;
    let warn = null;
    const o = originOf(v.base_url);
    if (o && o.startsWith("http://") && !isLocalHost(new URL(o).host))
      warn = status("warn", "Plain http to another machine: prompts travel unencrypted.");
    if (locked("base_url")) return h("div", null, label("endpoint", "base_url"), lockBox("base_url", v.base_url), reach);
    return h("div", null, label("endpoint", "base_url"),
      h("input", { id: "asst-url", class: "input", value: v.base_url, spellcheck: "false", autocomplete: "off",
        placeholder: `e.g. ${cfg.defaults.base_url[v.backend] || "https://…/v1"}`,
        style: { fontFamily: "JetBrains Mono, monospace", fontSize: "13px" },
        oninput: (e) => { v.base_url = e.target.value; keyEdit = false; confirmRemove = false; drawKey(); },
        onchange: () => refresh() }),
      reach, warn, h("p", { class: "field-help" }, URL_HELP[v.backend] || ""));
  }

  function keyField() {
    drawKey();
    if (v.backend === "ollama" && !cfg.key.env) {
      const saved = cfg.key.origins.includes(originOf(v.base_url));
      return h("div", null, label("API key", "api_key"),
        h("details", { open: saved || keyEdit }, h("summary", { class: "field-help", style: { cursor: "pointer" } },
          "Ollama behind a proxy that asks for a key?"), h("div", { style: { marginTop: "8px" } }, keySlot)),
        h("p", { class: "field-help" }, "An Ollama on this machine needs no key."));
    }
    return h("div", null, label("API key", "api_key"), keySlot);
  }

  function drawKey() {
    clear(keySlot);
    const o = originOf(v.base_url);
    const box = (cls, ...kids) => h("div", { class: `asst-key${cls ? " " + cls : ""}` }, ...kids);
    if (!o) { keySlot.append(box("", "Enter a valid http(s) endpoint first.")); return; }
    const host = new URL(o).host;
    if (cfg.key.env) {
      keySlot.append(box("", "Using HAX_ASSISTANT_API_KEY from the server environment; it is sent to the chat endpoint."));
      return;
    }
    if (cfg.key.origins.includes(o) && !keyEdit) {
      keySlot.append(box("", h("div", { class: "asst-key-saved" },
        h("div", null, h("div", { style: { fontWeight: 500 } }, `Key saved for ${host}`),
          h("div", { class: "field-help", style: { margin: 0 } }, "Stored on the server in a private file readable only by your account. It is not shown again.")),
        confirmRemove
          ? h("div", { class: "row", style: { gap: "8px" } }, h("span", { class: "field-help", style: { margin: 0 } }, `Remove the key for ${host}?`),
              h("button", { class: "btn danger small", onclick: removeKey }, "Remove key"),
              h("button", { class: "btn secondary small", onclick: () => { confirmRemove = false; drawKey(); } }, "Cancel"))
          : h("div", { class: "row", style: { gap: "8px" } },
              h("button", { class: "btn secondary small", onclick: () => { keyEdit = true; drawKey(); } }, "Replace"),
              h("button", { class: "btn secondary small", onclick: () => { confirmRemove = true; drawKey(); } }, "Remove")))));
      return;
    }
    if (!cfg.key.can_send) {
      keySlot.append(box("blocked", "This page reached the GUI over plain http from another computer, so the server won't accept a key from it. "
        + "Open the GUI on the server, through an SSH tunnel or over https, or run ", h("code", null, "hax-assistant key set"), " on the server."));
      return;
    }
    if (o.startsWith("http://") && !isLocalHost(host)) {
      keySlot.append(box("blocked", "This endpoint uses plain http, so a key would travel unencrypted. Use an https:// endpoint to save a key for it."));
      return;
    }
    const input = h("input", { id: "asst-key", class: "input", type: "password", autocomplete: "off", spellcheck: "false",
      placeholder: `Paste the key for ${host}` });
    const err = h("div");
    const saveKey = async () => {
      const key = input.value.trim();
      input.value = "";
      if (!key) { err.replaceChildren(status("err", "Paste a key first.")); return; }
      try {
        const c = await api.putAssistantKey(v.base_url, key);
        cfg = { ...cfg, key: c.key }; keyEdit = false;
        message = status("ok", `Key saved for ${host}.`);
        render(); refresh();
      } catch (e) { err.replaceChildren(errorBox(e)); }
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") saveKey(); });
    const ambient = o === "https://api.openai.com" && cfg.key.openai_env
      ? h("p", { class: "field-help" }, "Without a saved key, OPENAI_API_KEY from the server environment is used (it is only sent to api.openai.com).") : null;
    keySlot.append(h("div", null,
      h("div", { class: "asst-row" }, input,
        h("button", { class: "btn small", onclick: saveKey }, "Save key"),
        keyEdit ? h("button", { class: "btn secondary small", onclick: () => { keyEdit = false; drawKey(); } }, "Cancel") : null),
      err, ambient,
      h("p", { class: "field-help" }, `${keyEdit ? "The new key replaces" : "No key saved for"} ${host}. It is saved on the server for this endpoint only and never shown again.`)));
  }

  async function removeKey() {
    try {
      const c = await api.deleteAssistantKey(v.base_url);
      cfg = { ...cfg, key: c.key }; confirmRemove = false;
      message = status("ok", "Key removed. Nothing is sent to that endpoint now.");
      render(); refresh();
    } catch (e) { message = errorBox(e); render(); }
  }

  function modelField(key) {
    const chat = key === "model";
    const name = chat ? "chat model" : "embedding model";
    if (locked(key)) return h("div", null, label(name, key), lockBox(key, v[key]));
    const res = lists && (chat ? lists.chat : lists.embed);
    const entries = res && res.ok ? (chat ? res.chat : res.embed) : [];
    const hidden = chat && res && res.ok ? res.hidden : [];
    const cur = v[key];
    const listed = entries.some((m) => m.id === cur) || (showHidden && hidden.includes(cur));
    const sel = h("select", { id: `asst-${key}`, class: "input", onchange: (e) => {
      other[key] = e.target.value === "__other";
      if (!other[key]) v[key] = e.target.value;
      render();
    } });
    if (cur && !listed && !other[key]) sel.append(h("option", { value: cur, selected: true }, `${cur} (not in the list)`));
    if (!cur && !other[key]) sel.append(h("option", { value: "", selected: true, disabled: true }, res ? "Choose a model" : "Test the connection to list models"));
    for (const m of entries) sel.append(h("option", { value: m.id, selected: !other[key] && m.id === cur }, m.id));
    if (showHidden && hidden.length) {
      sel.append(h("optgroup", { label: "Not chat models by name" }, ...hidden.map((id) =>
        h("option", { value: id, selected: !other[key] && id === cur }, id))));
    }
    sel.append(h("option", { value: "__other", selected: other[key] }, "Other…"));
    const parts = [label(name, key),
      h("div", { class: "asst-row" }, sel,
        h("button", { class: "btn-icon", title: "List the models again", onclick: refresh, disabled: loading }, icon("refresh", 16)))];
    if (other[key]) {
      parts.push(h("input", { id: `asst-${key}-other`, class: "input", style: { marginTop: "8px" }, value: cur,
        placeholder: "The model name, as the provider spells it", oninput: (e) => { v[key] = e.target.value.trim(); } }));
    }
    const m = !other[key] ? entries.find((x) => x.id === cur) : null;
    if (m && (m.size || m.caps.length)) {
      parts.push(h("div", { class: "asst-caps" }, m.size ? h("span", null, m.size) : null,
        ...m.caps.filter((c) => c !== "completion").map((c) => h("span", { class: "cap" }, c))));
    }
    if (res && !res.ok && !chat && v.embed_base_url) parts.push(status("err", res.error));
    if (chat && m && m.tier) {
      parts.push(h("p", { class: "field-help" }, m.tier === "free"
        ? `Free tier: the model drives its own tool loop. ${m.tier_reason}.`
        : `Guided tier: fixed code plans, the model fills in and explains. ${m.tier_reason}.`));
    }
    if (chat && hidden.length) {
      parts.push(h("button", { class: "asst-link", onclick: () => { showHidden = !showHidden; render(); } },
        showHidden ? "Hide the models that are not chat models" : `Show ${hidden.length} more (audio, image, moderation…)`));
    }
    if (!chat) parts.push(h("p", { class: "field-help" }, "Used to search the docs. Changing it re-embeds the docs once."));
    if (!chat && res && res.ok && !entries.length) {
      parts.push(status("warn", "This endpoint lists no embedding models, so docs search can't run. Take embeddings from this machine's Ollama instead."));
    }
    return h("div", null, ...parts);
  }

  function embedSourceField() {
    const local = cfg.defaults.base_url.ollama;
    if (locked("embed_base_url")) {
      return h("div", null, label("embeddings from", "embed_base_url"), lockBox("embed_base_url", v.embed_base_url || "the chat endpoint"));
    }
    const choice = v.embed_base_url ? "ollama" : "same";
    return h("div", null, label("embeddings from", "embed_base_url"),
      h("div", { class: "chat-seg", role: "group", "aria-label": "Embeddings from", style: { display: "inline-flex" } },
        ...[["same", "The chat endpoint"], ["ollama", "This machine's Ollama"]].map(([k, name]) => h("button", {
          class: choice === k ? "active" : null, "aria-pressed": String(choice === k),
          onclick: () => {
            if (choice === k) return;
            v.embed_base_url = k === "ollama" ? local : "";
            if (!locked("embed_model")) v.embed_model = k === "ollama" ? cfg.defaults.embed_model : "";
            other.embed_model = false;
            render(); refresh();
          },
        }, name))),
      h("p", { class: "field-help" }, choice === "ollama"
        ? `Docs search runs on ${local}: free, and the questions you search with stay on this machine.`
        : "Docs search uses the chat endpoint's embeddings."));
  }

  function docsField() {
    const d = cfg.docs;
    let now = null;
    if (d.kind === "online") {
      now = `Using the online docs${d.last_modified ? `, site updated ${new Date(d.last_modified).toLocaleDateString()}` : ""}`;
    } else if (d.kind === "local") now = `Using the folder ${d.where}`;
    else now = "Docs are off, or the folder is missing.";
    if (d.chunks != null && d.kind) now += ` · ${d.chunks} sections indexed`;
    if (locked("docs")) return h("div", null, label("documentation", "docs"), lockBox("docs", v.docs), status("", now));
    const mode = docsMode();
    const options = [
      ["auto", "Automatic", d.local_checkout ? `The hax-em-docs checkout at ${d.local_checkout}.` : "The online docs (no hax-em-docs checkout next to the assistant)."],
      ["online", "Online docs", "The published docs site, downloaded once and then daily."],
      ["local", "Local folder", "A hax-em-docs checkout, for writing docs."],
      ["off", "Off", "The assistant answers from the program schema only and cites no pages."],
    ];
    const pathInput = h("input", { id: "asst-docs-path", class: "input", value: mode === "local" ? v.docs : "",
      placeholder: "/path/to/hax-em-docs/docs", oninput: (e) => { v.docs = e.target.value.trim(); } });
    const radios = h("div", { class: "asst-radios", role: "radiogroup", "aria-label": "Documentation" },
      ...options.map(([k, name, help]) => h("label", null,
        h("input", { type: "radio", name: "asst-docs", value: k, checked: mode === k,
          onchange: () => { v.docs = k === "local" ? (d.kind === "local" ? d.where : "") : k; docsMsg = null; render(); } }),
        h("span", null, name, h("small", null, help)))));
    const sub = mode === "local" ? h("div", { class: "asst-sub path-row" }, pathInput,
      h("button", { class: "btn-icon", title: "Browse server", onclick: () => openFilePicker("directory", (p) => {
        pathInput.value = p; v.docs = p; }, pathInput.value) }, icon("folderOpen", 16))) : null;
    return h("div", null, label("documentation", "docs"), radios, sub,
      h("div", { class: "asst-row", style: { marginTop: "10px" } }, status("", now),
        d.kind ? h("button", { class: "btn secondary small", onclick: checkDocs }, "Check for updates") : null),
      docsMsg);
  }

  async function checkDocs() {
    docsMsg = status("", "Checking…"); render();
    try {
      const r = await api.assistantCheckDocs();
      docsMsg = r.ok ? status("ok", r.rebuilt ? `Updated: ${r.chunks} sections indexed.` : `Up to date: ${r.chunks} sections.`)
        : status("err", r.reason);
      cfg = await api.assistantConfig();
    } catch (e) { docsMsg = errorBox(e); }
    render();
  }

  async function refresh() {
    if (!originOf(v.base_url)) return;
    loading = true; render();
    try {
      lists = await api.assistantModels({ backend: v.backend, base_url: v.base_url, embed_base_url: v.embed_base_url || "" });
      for (const [key, res, kind] of [["model", lists.chat, "chat"], ["embed_model", lists.embed, "embed"]]) {
        if (!v[key] && !locked(key) && res.ok && res[kind].length) v[key] = res[kind][0].id;
      }
    } catch (e) { message = errorBox(e); }
    loading = false; render();
  }

  async function save() {
    const body = {};
    for (const k of FIELDS) if (!locked(k)) body[k] = (v[k] || "").trim();
    if (docsMode() === "local" && !body.docs) { message = status("err", "Choose the docs folder first."); render(); return; }
    if (!locked("base_url") && !originOf(body.base_url)) { message = status("err", "Enter the endpoint URL first."); render(); return; }
    if (!locked("model") && !body.model) { message = status("err", "Choose a chat model first."); render(); return; }
    try {
      take(await api.putAssistantConfig(body));
      message = status("ok", `Saved. The next message uses ${cfg.values.model}; open chats keep their history.`);
    } catch (e) { message = errorBox(e); }
    render();
  }
}
