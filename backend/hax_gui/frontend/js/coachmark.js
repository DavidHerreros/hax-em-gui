// A single spotlight over one element: a scrim with a hole punched in it, plus a
// small card beside it.
//
// Deliberately *not* a tour. A multi-step tour is a selector into live markup for
// every step, plus an assumption about which panels are open when it runs — and
// when a step's target moves, nothing fails loudly, the tour just quietly points at
// the wrong thing. It also fires once, before the user has a project or a reason to
// care, and skipping it is permanent.
//
// One mark, fired from a button the user pressed, has none of that: it aims at
// something the caller has just made sure is on screen, and if the target is gone it
// simply doesn't appear (see the null return). The getting-started checklist uses it
// for the one thing a checklist genuinely cannot do — point at the collapsed
// programs rail, which is the piece of the shell people miss.
import { h } from "./h.js";

const PAD = 6;
const CARD_W = 262;

/**
 * @param target  Element, or a selector resolved against the document.
 * @param opts    { title, text, okText }
 * @returns a dismiss function, or null if the target isn't on screen.
 */
export function coachmark(target, { title, text, okText = "Got it" } = {}) {
  const el = typeof target === "string" ? document.querySelector(target) : target;
  if (!el || !el.getBoundingClientRect) return null;
  const first = el.getBoundingClientRect();
  if (!first.width || !first.height) return null;   // rendered but collapsed to nothing

  const hole = h("div", { class: "cm-hole" });
  const ok = h("button", { class: "btn", onclick: () => close() }, okText);
  const card = h("div", { class: "cm-card", role: "dialog", "aria-label": title || "Tip" },
    h("h4", null, title || ""),
    h("p", null, text || ""),
    h("div", { class: "cm-foot" }, ok));
  const layer = h("div", { class: "cm-layer" }, hole, card);
  document.body.appendChild(layer);
  place();
  ok.focus();

  // The click that opened this mark is still propagating, so wait a frame before
  // listening for the one that closes it.
  let armed = false;
  requestAnimationFrame(() => { armed = true; });

  const onDown = (e) => { if (armed && !card.contains(e.target)) close(); };
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  // Any layout change invalidates the measurements, and a mark left behind over
  // moved content is worse than no mark at all.
  const onMove = () => place();
  document.addEventListener("pointerdown", onDown, true);
  document.addEventListener("keydown", onKey, true);
  window.addEventListener("resize", onMove);
  window.addEventListener("scroll", onMove, true);
  window.addEventListener("hashchange", close);

  function place() {
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return close();      // target went away under us
    Object.assign(hole.style, {
      top: r.top - PAD + "px", left: r.left - PAD + "px",
      width: r.width + PAD * 2 + "px", height: r.height + PAD * 2 + "px",
    });
    // Sit on whichever side has room, then clamp to the viewport.
    const roomLeft = r.left, roomRight = window.innerWidth - r.right;
    const ch = card.offsetHeight || 150;
    const left = roomRight >= CARD_W + 20 || roomRight >= roomLeft ? r.right + 14 : r.left - CARD_W - 14;
    Object.assign(card.style, {
      left: Math.max(10, Math.min(left, window.innerWidth - CARD_W - 10)) + "px",
      top: Math.max(10, Math.min(r.top + r.height / 2 - ch / 2, window.innerHeight - ch - 10)) + "px",
    });
  }

  function close() {
    layer.remove();
    document.removeEventListener("pointerdown", onDown, true);
    document.removeEventListener("keydown", onKey, true);
    window.removeEventListener("resize", onMove);
    window.removeEventListener("scroll", onMove, true);
    window.removeEventListener("hashchange", close);
  }

  return close;
}
