"""Program registry: loads the cached form-schemas and groups programs.

Schemas are produced by :mod:`hax_gui.introspect`. We prefer the cached JSON
(fast, no JAX import) and fall back to live extraction if the cache is missing.
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import settings

# Human-facing category per hax subpackage, plus display order.
_CATEGORY_BY_PREFIX = [
    ("hax.networks", "Networks"),
    ("hax.programs", "Post-processing"),
    ("hax.metrics", "Metrics"),
    ("hax.viewers", "Viewers"),
]
_CATEGORY_ORDER = ["Networks", "Post-processing", "Metrics", "Viewers", "Other"]


def _category(module_path: str) -> str:
    for prefix, label in _CATEGORY_BY_PREFIX:
        if module_path.startswith(prefix):
            return label
    return "Other"


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    if settings.schema_cache.exists():
        return json.loads(settings.schema_cache.read_text())
    # Cache missing: extract live (imports JAX; slow but correct).
    from .introspect import extract_all

    data = extract_all()
    settings.schema_cache.parent.mkdir(parents=True, exist_ok=True)
    settings.schema_cache.write_text(json.dumps(data, indent=2, default=str))
    return data


def refresh() -> None:
    """Drop the in-memory cache (e.g. after regenerating schemas)."""
    _load.cache_clear()
    get_schema.cache_clear()


def all_programs() -> dict[str, Any]:
    return _load().get("programs", {})


# Extra form fields not discoverable by argparse introspection. annotate_space
# accepts arbitrary ``--key value`` kwargs forwarded to its server-functions
# module; ``--pickled_nn`` is the trained network the viewer decodes volumes
# from, so we surface it as a real (bindable) field.
_EXTRA_FIELDS: dict[str, list[dict[str, Any]]] = {
    "annotate_space": [
        {
            "name": "pickled_nn",
            "flags": ["--pickled_nn"],
            "flag": "--pickled_nn",
            "required": False,
            "help": "Trained network (HetSIREN / Zernike3Deep / ReconSIREN) the viewer uses to "
                    "decode volumes from latent coordinates. Bind a training run, or leave empty "
                    "for view-only. Needed when a server-functions module is set.",
            "default": None,
            "type": "string",
            "widget": "path",
            "path_kind": "directory",
        },
    ],
}


@lru_cache(maxsize=None)
def get_schema(name: str) -> dict[str, Any] | None:
    """Cached form-schema, augmented with GUI-only field defaults and extra
    (non-introspectable) fields. Returned read-only; callers must not mutate it."""
    raw = all_programs().get(name)
    if raw is None:
        return None
    extras = _EXTRA_FIELDS.get(name, [])
    overrides = field_defaults(name)
    if not extras and not overrides:
        return raw
    fields = [
        {**f, "default": overrides[f["name"]]} if f["name"] in overrides else f
        for f in raw["fields"]
    ]
    present = {f["name"] for f in fields}
    for extra in extras:
        if extra["name"] not in present:
            fields.append(dict(extra))
    return {**raw, "fields": fields}


def list_programs() -> list[dict[str, Any]]:
    """Lightweight catalog entries (no fields), sorted by category then name."""
    out = []
    for name, schema in all_programs().items():
        out.append(
            {
                "name": name,
                "description": schema.get("description", ""),
                "category": _category(schema.get("module", "")),
                "field_count": len(schema.get("fields", [])),
            }
        )
    out.sort(key=lambda p: (_CATEGORY_ORDER.index(p["category"]) if p["category"] in _CATEGORY_ORDER else 99, p["name"]))
    return out


def errors() -> dict[str, str]:
    return _load().get("errors", {})


# The output-directory argument each program writes to. The project manager
# auto-fills this with a per-run folder, so users never set it themselves.
_OUTPUT_ARG_OVERRIDES = {"annotate_space": "path"}
_DEFAULT_OUTPUT_ARG = "output_path"


def output_arg(name: str) -> str | None:
    """Name of the program's output-dir arg, or None if it has none."""
    schema = get_schema(name)
    if schema is None:
        return None
    arg = _OUTPUT_ARG_OVERRIDES.get(name, _DEFAULT_OUTPUT_ARG)
    return arg if any(f["name"] == arg for f in schema["fields"]) else None


# The SSD scratch-dir argument, auto-filled per project+program when a scratch
# root is configured, so users never set it themselves.
_SSD_ARG = "ssd_scratch_folder"


def ssd_arg(name: str) -> str | None:
    """Name of the program's SSD scratch-dir arg, or None if it has none."""
    schema = get_schema(name)
    if schema is None:
        return None
    return _SSD_ARG if any(f["name"] == _SSD_ARG for f in schema["fields"]) else None


# Metadata (.star/.xmd) files reference image stacks by relative paths, resolved
# against the process working directory. We launch each program from the folder
# of this argument so those relative references work regardless of where the
# server runs. Default is the particle metadata arg `md`.
_WORKDIR_ARG_OVERRIDES: dict[str, str] = {}
_DEFAULT_WORKDIR_ARG = "md"


def working_dir_arg(name: str) -> str | None:
    """Name of the arg whose folder should be the launch cwd, or None."""
    schema = get_schema(name)
    if schema is None:
        return None
    arg = _WORKDIR_ARG_OVERRIDES.get(name, _DEFAULT_WORKDIR_ARG)
    return arg if any(f["name"] == arg for f in schema["fields"]) else None


@lru_cache(maxsize=1)
def _hax_package_dir() -> Path | None:
    """Locate the installed ``hax`` package directory without importing it (so we
    don't pull in JAX just to resolve a bundled file path)."""
    try:
        spec = importlib.util.find_spec("hax")
    except (ImportError, ValueError):
        return None
    if spec and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    return None


def field_defaults(name: str) -> dict[str, Any]:
    """Per-field default values the GUI pre-fills into the form (editable).

    For ``annotate_space`` we point ``--server_functions_path`` at hax's bundled
    ``load_model.py`` so the user doesn't have to find/type it; they can still
    clear or override it (e.g. for view-only mode or a custom interface).
    """
    out: dict[str, Any] = {}
    if name == "annotate_space":
        pkg = _hax_package_dir()
        if pkg is not None:
            path = pkg / "viewers" / "server_loading_functions" / "load_model.py"
            if path.exists():
                out["server_functions_path"] = str(path)
    return out


