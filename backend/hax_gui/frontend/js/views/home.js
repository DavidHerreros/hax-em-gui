// Landing page: logo hero, documentation links, and the getting-started checklist.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { UpdateBanner } from "../updates.js";
import { SetupNotice } from "./setupNotice.js";
import { ResumeNotice } from "./resumeNotice.js";
import { GettingStarted } from "./gettingStarted.js";

const DOCS = "https://davidherreros.github.io/hax-em-docs/";

const DOC_CARDS = [
  { icon: "rocket", title: "Getting started", desc: "Install Hax, set up your environment, and learn the core concepts behind conformational heterogeneity analysis.", href: DOCS + "getting-started/" },
  { icon: "fileText", title: "Programs reference", desc: "What each tool does and every parameter it accepts — the same options the GUI generates its forms from.", href: DOCS + "programs/" },
  { icon: "workflow", title: "Workflows & examples", desc: "End-to-end pipelines and real-case walkthroughs combining several programs.", href: DOCS + "workflows/" },
];

export function HomeView() {
  return h(
    "div",
    { class: "view" },
    UpdateBanner(),
    SetupNotice(),
    ResumeNotice(),
    hero(),
    // Before the docs on purpose: "what do I do next" outranks "what is there to read".
    GettingStarted(),
    section("Documentation", h("div", { class: "grid" }, ...DOC_CARDS.map(docCard)))
  );
}

function hero() {
  return h(
    "div",
    { class: "hero" },
    h("div", { class: "logo-tile" }, h("img", { src: "/assets/logo.png", alt: "Hax logo" })),
    h(
      "div",
      null,
      h("h1", null, "Hax"),
      h("p", { class: "tagline" }, "A suite of tools for studying conformational heterogeneity in Cryo-EM data with JAX-powered neural networks — now driveable from your browser, against a remote GPU server."),
      h(
        "div",
        { class: "hero-actions" },
        h("a", { class: "btn", href: "#/projects" }, icon("box", 18), "Go to Projects"),
        h("a", { class: "btn secondary", href: DOCS, target: "_blank", rel: "noreferrer" }, icon("book", 18), "Open documentation")
      )
    )
  );
}

function section(title, body) {
  return h("section", { style: { marginBottom: "8px" } }, h("h2", { class: "section" }, title), body);
}

function docCard(c, i) {
  return h(
    "a",
    { class: "feature", href: c.href, target: "_blank", rel: "noreferrer", style: { animationDelay: `${i * 0.04}s` } },
    ficon(c.icon),
    h("h3", null, c.title),
    h("p", null, c.desc),
    h("span", { class: "ext" }, "Read the docs", icon("external", 14))
  );
}

function ficon(name) {
  const box = h("div", { class: "ficon" });
  box.appendChild(icon(name, 20));
  return box;
}
