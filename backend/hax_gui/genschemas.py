"""Regenerate the cached program form-schemas (imports JAX via introspect).

Used by both the ``hax-gui schemas`` CLI subcommand and the dev helper script
``backend/scripts/generate_schemas.py``.
"""

from __future__ import annotations

import json
from typing import Any

from .config import settings


def regenerate(seed: bool = False) -> dict[str, Any]:
    """Extract program schemas and write them to the cache.

    With ``seed=False`` (default) writes the per-user cache under the data dir
    (writable on an installed system). With ``seed=True`` writes the cache
    bundled in the package — the shipped default, to be committed before a
    release. Returns the extraction result (``{"programs": ..., "errors": ...}``).
    """
    from .introspect import extract_all   # imports JAX; keep lazy

    target = settings.packaged_schema_cache if seed else settings.schema_cache
    result = extract_all()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, default=str))
    result["_target"] = str(target)
    return result
