"""Translate a program name + form values into a hax_project_manager argv.

Mirrors the CLI contract from hax/cli/cli.py:
    hax_project_manager [--gpu N] <program> --flag value ...
Global flags (``--gpu``) come *before* the program name.
"""

from __future__ import annotations

import shlex
from typing import Any

from .config import settings
from .registry import get_schema


class ValidationError(ValueError):
    """Raised when submitted values don't satisfy the program schema."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def strip_obsolete(program: str, values: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop values whose argument the program no longer has. Returns ``(kept, dropped)``.

    For *re-running a stored job* only — never for a fresh submission, where an unknown
    key means the form and the schema disagree and :func:`build_command` should say so.

    A job records the arguments it ran with, and a program's CLI moves on: an option gets
    renamed or removed, and every run that used it becomes impossible to re-run, because
    the stored value no longer corresponds to anything. Refusing was the honest answer but
    a dead end — the run could not be repeated from the canvas or the Jobs view at all,
    only rebuilt by hand through the form (which filters to the current fields and so
    quietly did this already). Dropping the obsolete values reproduces the run as closely
    as the program still allows, and the caller reports what went.
    """
    schema = get_schema(program)
    if schema is None:
        return dict(values), []
    fields = {f["name"] for f in schema["fields"]}
    dropped = sorted(k for k in values if k not in fields)
    return ({k: v for k, v in values.items() if k in fields}, dropped)


def build_command(program: str, values: dict[str, Any], gpu: str | None = None) -> list[str]:
    """Build the full argv list, validating required/typed fields first."""
    schema = get_schema(program)
    if schema is None:
        raise ValidationError({"program": f"unknown program '{program}'"})

    fields = {f["name"]: f for f in schema["fields"]}
    errors: dict[str, str] = {}

    # Reject unknown keys early -- they usually signal a frontend/schema drift.
    for key in values:
        if key not in fields:
            errors[key] = "unknown argument"

    # hax_cli is normally the single ``hax_project_manager`` binary, but allow a
    # multi-token launcher (e.g. ``python -m hax.cli``) via shlex splitting.
    argv: list[str] = shlex.split(settings.hax_cli)
    if gpu == "cpu":
        argv += ["--gpu", ""]  # CUDA_VISIBLE_DEVICES="" -> run on CPU
    elif gpu not in (None, ""):
        argv += ["--gpu", str(gpu)]
    argv.append(program)

    for name, field in fields.items():
        value = values.get(name)
        ftype = field["type"]

        if ftype == "boolean":
            # The form value is the *destination* value. For store_true the flag
            # produces True, so emit it when checked; for store_false (invert) the
            # flag produces False, so emit it when UNchecked — appending it for a
            # checked box would silently flip the user's choice.
            if bool(value) != bool(field.get("invert")):
                argv.append(field["flag"])
            continue

        if _is_empty(value):
            if field["required"]:
                errors[name] = "required"
            continue

        if ftype == "enum" and value not in field.get("choices", []):
            errors[name] = f"must be one of {field.get('choices')}"
            continue
        if ftype == "integer" and not _coerces(value, int):
            errors[name] = "must be an integer"
            continue
        if ftype == "number" and not _coerces(value, float):
            errors[name] = "must be a number"
            continue

        argv.append(field["flag"])
        if field.get("multiple") and isinstance(value, (list, tuple)):
            argv += [str(v) for v in value]
        else:
            argv.append(str(value))

    if errors:
        raise ValidationError(errors)
    return argv


def _coerces(value: Any, kind: type) -> bool:
    try:
        kind(value)
        return True
    except (TypeError, ValueError):
        return False
