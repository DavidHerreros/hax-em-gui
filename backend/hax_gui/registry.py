"""Program registry: loads the cached form-schemas and groups programs.

Schemas are produced by :mod:`hax_gui.introspect`. We prefer the cached JSON
(fast, no JAX import) and fall back to live extraction if the cache is missing.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import adapters, groups
from .config import settings

# Human-facing category per hax subpackage, plus display order. Matched in order,
# first hit wins, so a more specific prefix must precede the one it extends.
_CATEGORY_BY_PREFIX = [
    ("hax.preprocessing", "Preprocessing"),
    ("hax.networks", "Networks"),
    ("hax.programs", "Post-processing"),
    ("hax.metrics", "Metrics"),
    ("hax.viewers", "Viewers"),
]
# Preprocessing leads: it is the first thing you run on a project.
# Keep in sync with CATEGORY_ORDER in frontend/js/views/{launcher,workflows}.js.
_CATEGORY_ORDER = ["Preprocessing", "Networks", "Post-processing", "Metrics", "Viewers", "Other"]


def _category(module_path: str) -> str:
    for prefix, label in _CATEGORY_BY_PREFIX:
        if module_path.startswith(prefix):
            return label
    return "Other"


def _load_cached() -> dict[str, Any]:
    if settings.schema_cache.exists():
        return json.loads(settings.schema_cache.read_text())
    # No user cache yet: use the seed shipped inside the package, if present.
    if settings.packaged_schema_cache.exists():
        return json.loads(settings.packaged_schema_cache.read_text())
    # Nothing cached anywhere: extract live (imports JAX; slow but correct) and
    # persist into the (writable) user cache.
    from .introspect import extract_all

    data = extract_all()
    settings.schema_cache.parent.mkdir(parents=True, exist_ok=True)
    settings.schema_cache.write_text(json.dumps(data, indent=2, default=str))
    return data


def _declared_programs() -> dict[str, tuple[str, str]] | None:
    """Read hax's ``MODULES_DICT`` straight from ``hax/cli/cli.py`` source via
    ``ast`` — no ``import hax`` (which would pull in JAX). Returns ``{name:
    (module_path, description)}`` or ``None`` if hax isn't found / the dict isn't a
    plain literal (in which case auto-discovery quietly no-ops)."""
    pkg = _hax_package_dir()
    if pkg is None:
        return None
    cli = pkg / "cli" / "cli.py"
    try:
        tree = ast.parse(cli.read_text())
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MODULES_DICT" for t in node.targets
        ):
            try:
                raw = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
            return {k: (v[0], v[1]) for k, v in raw.items()
                    if isinstance(v, (list, tuple)) and len(v) >= 2}
    return None


def _autodiscover(data: dict[str, Any]) -> None:
    """Reconcile the loaded schema cache with hax's current ``MODULES_DICT``:
    live-introspect programs that were added since the cache was built and drop
    ones that were removed, then persist to the user cache. This keeps the GUI in
    step with new hax programs automatically — the only manual step left is
    regenerating (``hax-gui schemas``) when an existing program's *arguments*
    change, which source inspection can't detect. Only imports JAX when a genuinely
    new program must be extracted."""
    declared = _declared_programs()
    if not declared:
        return
    progs: dict[str, Any] = data.setdefault("programs", {})
    errs: dict[str, Any] = data.setdefault("errors", {})
    missing = [n for n in declared if n not in progs and n not in errs]
    removed = [n for n in list(progs) if n not in declared]
    if not missing and not removed:
        return
    for name in removed:
        progs.pop(name, None)
    if missing:
        from .introspect import schema_for_program   # imports JAX; keep lazy

        for name in missing:
            module_path, desc = declared[name]
            try:
                progs[name] = schema_for_program(name, module_path, desc)
            except Exception as exc:  # noqa: BLE001 - record & skip, like extract_all
                errs[name] = str(exc)
    try:
        settings.schema_cache.parent.mkdir(parents=True, exist_ok=True)
        settings.schema_cache.write_text(json.dumps(data, indent=2, default=str))
    except OSError:
        pass


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    data = _load_cached()
    _autodiscover(data)
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
    """Cached form-schema, augmented with GUI-only field defaults, extra
    (non-introspectable) fields, and the ``group``/``advanced`` annotations the
    form groups by. Returned read-only; callers must not mutate it.

    Grouping is applied here rather than baked into the schema cache, so tuning
    the rules in :mod:`hax_gui.groups` takes effect without regenerating schemas.
    """
    raw = all_programs().get(name)
    if raw is None:
        return None
    extras = _EXTRA_FIELDS.get(name, [])
    overrides = field_defaults(name)
    fields = [
        {**f, "default": overrides[f["name"]]} if f["name"] in overrides else f
        for f in raw["fields"]
    ]
    present = {f["name"] for f in fields}
    for extra in extras:
        if extra["name"] not in present:
            fields.append(dict(extra))
    return {**raw, "fields": groups.annotate(fields)}


def _has_train_predict_mode(schema: dict[str, Any]) -> bool:
    """True if the program exposes a ``--mode`` with both ``train`` and ``predict``
    (so a train→predict wiring can inherit the training run's parameters)."""
    for f in schema.get("fields", []):
        if f.get("name") == "mode":
            choices = set(f.get("choices") or [])
            return "train" in choices and "predict" in choices
    return False


# hax entry points that are not batch jobs, so they must not appear in the launcher
# or the workflow palette. Both would be actively wrong as a job:
#   display_metrics — starts TensorBoard and blocks forever; as a workflow step it
#     would never terminate and its dependents would never run. The GUI reads the
#     same logs natively (see the Metrics tab and hax_gui/tensorboard.py).
#   update — runs `pip install -U` / `git pull` and prompts for confirmation. The GUI
#     has its own Updater (updater.py) with restart handling; running this as a job
#     would upgrade hax underneath the server that launched it.
# Only the *catalog* is filtered. `get_schema` and friends still resolve these names,
# so a job or workflow node saved before this change keeps working.
_NOT_A_JOB = frozenset({"display_metrics", "update"})


def list_programs() -> list[dict[str, Any]]:
    """Lightweight catalog entries (no fields), sorted by category then name.

    Programs in :data:`_NOT_A_JOB` are omitted — this list feeds the launcher and the
    workflow palette, which offer to *run* what they show.

    ``moded`` (has a train/predict ``--mode``) and ``latents_decoder`` (a predict
    run yields latents + a reloadable decoder) are derived flags so the frontend
    can group programs without hardcoding their names — new programs registered in
    :mod:`hax_gui.adapters` are picked up automatically."""
    latents_decoders = set(adapters.latents_decoder_programs())
    out = []
    for name, schema in all_programs().items():
        if name in _NOT_A_JOB:
            continue
        out.append(
            {
                "name": name,
                "description": schema.get("description", ""),
                "category": _category(schema.get("module", "")),
                "field_count": len(schema.get("fields", [])),
                "moded": _has_train_predict_mode(schema),
                "latents_decoder": name in latents_decoders,
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


