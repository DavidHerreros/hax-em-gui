#!/usr/bin/env python3
"""Cross-file check that every relative ES-module import resolves to a real
file and a real export.

``node --check`` validates each file in isolation, so a syntactically fine
``import { x } from "./api.js"`` naming an export that does not exist passes
every per-file check — and then kills instantiation of the *entire* module
graph in the browser (blank page, one console error, zero network calls
afterwards). This happened once (events.js importing a non-exported helper);
this script makes it impossible to ship again.

Usage: check_js_imports.py <dir> [<dir> ...]   (exits 1 on any violation)
"""

from __future__ import annotations

import os
import re
import sys

IMPORT_RE = re.compile(
    r'import\s+(?:(\w+)\s*,?\s*)?(?:\{([^}]*)\})?\s*from\s*["\']([^"\']+)["\']')


def exports_of(src: str) -> set[str]:
    names: set[str] = set()
    for m in re.finditer(r'export\s+(?:async\s+)?(?:function|class)\s+(\w+)', src):
        names.add(m.group(1))
    for m in re.finditer(r'export\s+(?:const|let|var)\s+(\w+)', src):
        names.add(m.group(1))
    for m in re.finditer(r'export\s*\{([^}]*)\}', src):
        for part in m.group(1).split(","):
            part = part.strip()
            if part:
                names.add(part.split(" as ")[-1].strip())
    if re.search(r'export\s+default', src):
        names.add("default")
    return names


def check_tree(root: str) -> list[str]:
    files: dict[str, str] = {}
    for dirpath, _dirs, fnames in os.walk(root):
        for fn in fnames:
            if fn.endswith(".js"):
                path = os.path.join(dirpath, fn)
                files[path] = open(path, encoding="utf8").read()

    bad: list[str] = []
    for path, src in files.items():
        rel = os.path.relpath(path, root)
        for m in IMPORT_RE.finditer(src):
            default_imp, named, spec = m.groups()
            if not spec.startswith("."):
                continue  # bare specifiers (none expected, but not ours to judge)
            target = os.path.normpath(os.path.join(os.path.dirname(path), spec))
            if target not in files:
                bad.append(f"{rel}: import of missing file {spec!r}")
                continue
            exp = exports_of(files[target])
            if default_imp and "default" not in exp:
                bad.append(f"{rel}: {spec} has no default export "
                           f"(import {default_imp})")
            for n in (named or "").split(","):
                n = n.strip()
                if not n:
                    continue
                orig = n.split(" as ")[0].strip()
                if orig not in exp:
                    bad.append(f"{rel}: {spec} does not export {orig!r}")
    return bad


def main() -> int:
    roots = sys.argv[1:]
    if not roots:
        here = os.path.dirname(os.path.abspath(__file__))
        pkg = os.path.normpath(os.path.join(here, "..", "hax_gui"))
        roots = [os.path.join(pkg, "frontend"),
                 os.path.join(pkg, "viewer_app", "static")]
    bad: list[str] = []
    for root in roots:
        bad.extend(check_tree(root))
    if bad:
        print("\n".join(bad))
        return 1
    print(f"js imports OK across {len(roots)} tree(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
