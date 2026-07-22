// The parameter form shared by the right-menu launcher and the workflow node
// inspector. One component, two configurations — before this, each surface had
// its own renderer and they had already drifted (different boolean widgets, no
// CSV/path hints in the inspector).
//
// What it adds over a flat field list: accordion groups (from `field.group`,
// ordered by the schema's `group_order`), a search box, and an advanced tier
// (`field.advanced`) hidden behind a switch. `hetsiren` has 51 settable
// parameters; 27 of them are basic, which is the difference between a form you
// can read in a 264 px panel and one you cannot.
import { h, clear } from "../h.js";
import { icon } from "../icons.js";
import { fieldControl, isModified } from "./formFields.js";

const ADV_KEY = "hax_gui_show_advanced";   // global preference, not per program

// Which sections are open, and what is typed in the search box, keyed by program.
// Both surfaces rebuild their form on unrelated events (the inspector re-renders
// on every node selection, the launcher on every prefill), so this state has to
// outlive the component or it would reset under the user mid-edit.
const uiState = {};
const stateFor = (program) => (uiState[program] ??= { open: null, query: "" });

const readAdvanced = () => {
  try { return localStorage.getItem(ADV_KEY) === "1"; } catch { return false; }
};
const writeAdvanced = (on) => {
  try { localStorage.setItem(ADV_KEY, on ? "1" : "0"); } catch { /* private mode */ }
};

// Search matches the flag, the group and the help text, ranked so exact-ish name
// hits come first. Deliberately not fuzzy: on 50 long, descriptive identifiers
// subsequence matching mostly produces confident nonsense.
function score(field, q) {
  const name = field.name.toLowerCase();
  if (name.startsWith(q)) return 4;
  if (name.includes(q) || field.flag.toLowerCase().includes(q)) return 3;
  if ((field.group || "").toLowerCase().includes(q)) return 2;
  if ((field.help || "").toLowerCase().includes(q)) return 1;
  return 0;
}

// Wrap the first occurrence of `q` in a <mark>, so a hit in a long help string is
// findable without re-reading it.
function highlight(text, q) {
  if (!q) return text;
  const i = text.toLowerCase().indexOf(q);
  if (i < 0) return text;
  return h("span", null, text.slice(0, i), h("mark", null, text.slice(i, i + q.length)), text.slice(i + q.length));
}

/**
 * Build the form.
 *
 * @param schema  program schema from `/api/programs/{name}` (fields carry
 *                `group` / `advanced`; the schema carries `group_order`)
 * @param fields  the fields to render, already filtered (see `visibleFields`)
 * @param opts
 *   get(name)             read a value
 *   set(name, v, field)    write one — the caller owns coercion and whether an
 *                          empty value is stored or deleted (dense launcher
 *                          values vs. sparse workflow-node overrides)
 *   density   "comfortable" | "compact"
 *   intro     node rendered above the search bar (run name, notes, …)
 *   override(field)             node used INSTEAD of the input (a wired edge)
 *   augment(field, ctl, row)    node appended after the input (a binding control)
 *   tag(field)                  short chip shown next to the flag
 * @returns { el, setErrors, clearErrors }
 */
export function paramForm(schema, fields, opts = {}) {
  const { get, set, density = "comfortable", intro = null } = opts;
  const override = opts.override || (() => null);
  const augment = opts.augment || (() => null);
  const tag = opts.tag || (() => null);

  const ui = stateFor(schema.name);
  let showAdvanced = readAdvanced();
  let errors = {};

  const order = schema.group_order || [];
  const orderKey = (g) => { const i = order.indexOf(g); return i < 0 ? order.length : i; };
  // Groups in schema order, each with its fields in declaration order.
  const allGroups = [...new Set(fields.map((f) => f.group || "General"))]
    .sort((a, b) => orderKey(a) - orderKey(b) || a.localeCompare(b))
    .map((name) => ({ name, fields: fields.filter((f) => (f.group || "General") === name) }));
  if (ui.open === null) ui.open = new Set(allGroups.length ? [allGroups[0].name] : []);

  const advancedCount = fields.filter((f) => f.advanced).length;

  // -- chrome (built once; never re-rendered, so typing never loses focus) ----
  const searchInput = h("input", {
    class: "input", type: "search", placeholder: "Search parameters…",
    "aria-label": "Search parameters", value: ui.query,
    oninput: (e) => { ui.query = e.target.value; renderGroups(); },
  });
  const clearBtn = h("button", {
    class: "pf-clear", type: "button", title: "Clear search",
    onclick: () => { ui.query = ""; searchInput.value = ""; renderGroups(); searchInput.focus(); },
  }, icon("x", 13));
  const advSwitch = h("button", {
    class: "toggle small" + (showAdvanced ? " on" : ""), type: "button", role: "switch",
    "aria-checked": String(showAdvanced), "aria-label": "Show advanced parameters",
  }, h("span", { class: "knob" }));
  const advCount = h("span", { class: "pf-adv-count mono" });
  advSwitch.onclick = () => {
    showAdvanced = !showAdvanced;
    writeAdvanced(showAdvanced);
    advSwitch.classList.toggle("on", showAdvanced);
    advSwitch.setAttribute("aria-checked", String(showAdvanced));
    renderGroups();
  };

  const chrome = h("div", { class: "pf-chrome" },
    h("div", { class: "pf-search" }, icon("search", 14), searchInput, clearBtn),
    advancedCount
      ? h("div", { class: "pf-adv" }, advSwitch,
          h("span", null, "Advanced parameters"), advCount)
      : null);

  const groupsEl = h("div", { class: "pf-groups" });
  const el = h("div", { class: "pf pf-" + density }, intro, chrome, groupsEl);

  // -- rows -------------------------------------------------------------------
  function fieldRow(field, q, showGroup) {
    const err = h("p", { class: "field-error" }, errors[field.name] || "");
    const chips = [];
    if (showGroup) chips.push(h("span", { class: "pf-chip grp" }, field.group));
    if (field.advanced) chips.push(h("span", { class: "pf-chip adv" }, "advanced"));
    const extra = tag(field);
    if (extra) chips.push(h("span", { class: "pf-chip" }, extra));
    if (field.custom_type) chips.push(h("span", { class: "ctype" }, field.custom_type));

    const row = h("div", { class: "field" + (errors[field.name] ? " has-error" : "") },
      h("div", { class: "field-label" },
        h("span", { class: "flag" }, highlight(field.flag, q),
          field.required ? h("span", { class: "req" }, "*") : null),
        chips.length ? h("span", { class: "pf-chips" }, chips) : null));
    row.dataset.field = field.name;

    const replaced = override(field);
    const control = replaced || fieldControl(field, () => get(field.name), (v) => {
      set(field.name, v, field);
      // Clear a stale server-side error as soon as the field is touched.
      if (errors[field.name]) { delete errors[field.name]; err.textContent = ""; row.classList.remove("has-error"); }
      refreshCounts();
    });
    row.appendChild(control);

    if (!replaced) {
      const after = augment(field, control, row);
      if (after) row.appendChild(after);
    }
    if (field.help) row.appendChild(h("p", { class: "field-help" }, highlight(field.help, q)));
    row.appendChild(err);
    return row;
  }

  // -- group / search rendering ----------------------------------------------
  const modifiedIn = (list) => list.filter((f) => isModified(f, get(f.name))).length;

  function renderGroups() {
    clear(groupsEl);
    const q = ui.query.trim().toLowerCase();
    clearBtn.style.visibility = q ? "visible" : "hidden";
    if (advancedCount) {
      advCount.textContent = showAdvanced
        ? `${fields.length} shown`
        : `${advancedCount} hidden`;
    }

    if (q) return renderSearch(q);

    for (const g of allGroups) {
      const list = g.fields.filter((f) => showAdvanced || !f.advanced);
      if (!list.length) continue;
      const open = ui.open.has(g.name);
      const changed = modifiedIn(list);
      const head = h("button", {
        class: "pf-head" + (open ? " open" : ""), type: "button", "aria-expanded": String(open),
        onclick: () => { open ? ui.open.delete(g.name) : ui.open.add(g.name); renderGroups(); },
      },
        h("span", { class: "pf-chev" }, icon("chevronRight", 12)),
        h("span", { class: "pf-head-name" }, g.name),
        changed ? h("span", { class: "pf-badge mono", title: `${changed} changed from the default` }, String(changed)) : null,
        h("span", { class: "pf-head-n mono" }, String(list.length)));
      groupsEl.appendChild(head);
      if (open) groupsEl.appendChild(h("div", { class: "pf-fields" }, ...list.map((f) => fieldRow(f, "", false))));
    }
  }

  function renderSearch(q) {
    // Search always reaches into hidden advanced parameters — a search box that
    // silently omits matches is worse than none. They're chipped instead.
    const hits = fields
      .map((f) => ({ f, s: score(f, q) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s || a.f.name.localeCompare(b.f.name))
      .map((x) => x.f);
    const hidden = showAdvanced ? 0 : hits.filter((f) => f.advanced).length;
    groupsEl.appendChild(h("p", { class: "pf-note" },
      hits.length ? `${hits.length} match${hits.length === 1 ? "" : "es"} across all groups` : "No parameter matches that.",
      hidden ? ` · including ${hidden} advanced` : ""));
    if (hits.length) {
      groupsEl.appendChild(h("div", { class: "pf-fields" }, ...hits.map((f) => fieldRow(f, q, true))));
    }
  }

  // Keep the per-group "changed" badges honest without re-rendering the fields
  // (which would blur whatever input the user is typing into).
  function refreshCounts() {
    const heads = groupsEl.querySelectorAll(".pf-head");
    let i = 0;
    for (const g of allGroups) {
      const list = g.fields.filter((f) => showAdvanced || !f.advanced);
      if (!list.length) continue;
      const head = heads[i++];
      if (!head) continue;
      const changed = modifiedIn(list);
      const badge = head.querySelector(".pf-badge");
      if (changed && badge) badge.textContent = String(changed);
      else if (changed) head.insertBefore(h("span", { class: "pf-badge mono" }, String(changed)), head.querySelector(".pf-head-n"));
      else if (badge) badge.remove();
    }
  }
  el.addEventListener("change", refreshCounts);

  renderGroups();

  return {
    el,
    /** Show server-side (422) validation errors, opening and revealing the first
     *  offending field — otherwise an error inside a collapsed group is invisible. */
    setErrors(map) {
      errors = { ...map };
      const names = Object.keys(errors);
      if (names.length) {
        ui.query = "";
        searchInput.value = "";
        for (const f of fields) if (errors[f.name]) ui.open.add(f.group || "General");
      }
      renderGroups();
      const first = names.length && groupsEl.querySelector(`.field[data-field="${CSS.escape(names[0])}"]`);
      if (first) first.scrollIntoView({ block: "center", behavior: "smooth" });
    },
    clearErrors() {
      if (!Object.keys(errors).length) return;
      errors = {};
      renderGroups();
    },
  };
}
