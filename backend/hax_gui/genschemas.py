"""Regenerate the cached program form-schemas (imports JAX via introspect).

Used by both the ``hax-gui schemas`` CLI subcommand and the dev helper script
``backend/scripts/generate_schemas.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .config import settings


def regenerate(seed: bool = False) -> dict[str, Any]:
    """Extract program schemas and write them to the cache.

    With ``seed=False`` (default) writes the per-user cache under the data dir
    (writable on an installed system). With ``seed=True`` writes the cache
    bundled in the package — the shipped default, to be committed before a
    release. Returns the extraction result (``{"programs": ..., "errors": ...}``).

    A program that fails to introspect *keeps the schema it had* in the cache being
    replaced, listed under ``kept`` with the error. The background regeneration runs
    whenever a hax source file changes, i.e. often in the middle of an edit — an
    import that is broken for a minute must not empty the launcher, the workflow
    palette and the viewer's notion of which runs have latents until the next edit
    happens to land on a good state (which is what used to happen, and stuck:
    programs under ``errors`` are not retried). ``errors`` is left for programs
    that have no schema at all, which is what the registry's own retry keys on.
    """
    from .introspect import extract_all   # imports JAX; keep lazy

    target = settings.packaged_schema_cache if seed else settings.schema_cache
    result = extract_all()
    result["kept"] = {}
    previous = _read(target)
    if previous:
        for name, err in list(result["errors"].items()):
            old = previous.get("programs", {}).get(name)
            if old is not None:
                result["programs"][name] = old
                result["kept"][name] = err
                del result["errors"][name]
    if result["kept"]:
        names = ", ".join(sorted(result["kept"]))
        first = next(iter(result["kept"].values()))
        print(f"[hax-gui] schemas: {len(result['kept'])} program(s) failed to introspect and keep "
              f"their previous schema ({names}); first error: {first}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: a half-written cache used to 500 every schema endpoint until deleted.
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str))
    os.replace(tmp, target)
    result["_target"] = str(target)
    return result


def _read(path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
