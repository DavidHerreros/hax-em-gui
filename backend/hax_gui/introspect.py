"""Introspect hax-em program argparse parsers into JSON form-schemas.

Every hax program is an argparse ``main()`` that builds its parser inline and
then calls ``parse_args()`` (see ``hax/cli/cli.py:MODULES_DICT``). We extract the
fully-built parser *without running the program* by monkeypatching
``ArgumentParser.parse_args`` to capture ``self`` and abort right after the
parser is constructed but before any real work happens.

The resulting schema drives the auto-generated forms in the web GUI, so a new or
changed hax program needs **zero** GUI code -- the form regenerates from here.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from typing import Any

# Strip the bcolors ANSI escape codes that hax help strings are full of.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Heuristics to tag path-like arguments so the frontend shows a server file
# picker instead of a plain text box. Refined per-program later if needed.
_DIR_HINTS = ("output_path", "scratch", "logdir", "_dir", "folder")
_DIR_EXACT = ("path",)
_FILE_HINTS = ("md", "vol", "mask", "z_space", "weights", "metadata", "_file", "reload")


class _CapturedParser(Exception):
    """Sentinel raised to unwind out of ``main()`` once the parser is built."""

    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser
        super().__init__("captured parser")


def _clean(text: str | None) -> str:
    return _ANSI_RE.sub("", text or "").strip()


def extract_parser(module_path: str, program_name: str) -> argparse.ArgumentParser:
    """Import ``module_path`` and capture the parser its ``main()`` builds."""
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        raise RuntimeError(f"{module_path} exposes no main()")

    originals = {
        "parse_args": argparse.ArgumentParser.parse_args,
        "parse_known_args": argparse.ArgumentParser.parse_known_args,
    }

    def _capture(self, *args, **kwargs):  # noqa: ANN001
        raise _CapturedParser(self)

    argparse.ArgumentParser.parse_args = _capture
    argparse.ArgumentParser.parse_known_args = _capture
    saved_argv = sys.argv
    sys.argv = [program_name]
    try:
        module.main()
    except _CapturedParser as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_args = originals["parse_args"]
        argparse.ArgumentParser.parse_known_args = originals["parse_known_args"]
        sys.argv = saved_argv
    raise RuntimeError(f"{program_name}: main() never called parse_args()")


def _path_widget(name: str) -> tuple[str, str] | None:
    """Return (widget, path_kind) if the argument looks like a server path."""
    low = name.lower()
    if low in _DIR_EXACT or any(h in low for h in _DIR_HINTS):
        return "path", "directory"
    if any(h in low for h in _FILE_HINTS) or "path" in low:
        return "path", "file"
    return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, list, dict)):
        return value
    return str(value)


def field_for_action(action: argparse.Action) -> dict[str, Any] | None:
    """Map one argparse action to a JSON form-field descriptor."""
    if action.dest == "help" or isinstance(action, argparse._HelpAction):
        return None

    field: dict[str, Any] = {
        "name": action.dest,
        "flags": list(action.option_strings),
        "flag": action.option_strings[-1] if action.option_strings else action.dest,
        "required": bool(getattr(action, "required", False)) or not action.option_strings,
        "help": _clean(action.help),
        "default": _json_safe(action.default),
    }

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        field["type"] = "boolean"
        field["default"] = bool(action.default)
    elif action.choices:
        field["type"] = "enum"
        field["choices"] = [_json_safe(c) for c in action.choices]
    elif action.type is int:
        field["type"] = "integer"
    elif action.type is float:
        field["type"] = "number"
    else:
        field["type"] = "string"
        custom = action.type
        if callable(custom) and getattr(custom, "__name__", "") not in ("str", "type", ""):
            # e.g. list_of_floats -> comma-separated entry
            field["custom_type"] = getattr(custom, "__name__", str(custom))
            field["widget"] = "csv"

    # Lists: nargs of '+'/'*' or a positive int. nargs==0 is a store_true flag.
    if action.nargs in ("+", "*") or (isinstance(action.nargs, int) and action.nargs > 0):
        field["multiple"] = True
        field["nargs"] = action.nargs

    if field["type"] == "string" and "widget" not in field:
        widget = _path_widget(action.dest)
        if widget:
            field["widget"], field["path_kind"] = widget

    return field


def schema_for_program(program_name: str, module_path: str, description: str) -> dict[str, Any]:
    """Build the full form-schema for one program."""
    parser = extract_parser(module_path, program_name)
    fields = [f for a in parser._actions if (f := field_for_action(a)) is not None]
    return {
        "name": program_name,
        "module": module_path,
        "description": _clean(description),
        "summary": _clean(parser.description),
        "fields": fields,
    }


def extract_all() -> dict[str, Any]:
    """Introspect every program in hax's MODULES_DICT.

    Programs that fail to import (e.g. GUI deps missing in a headless env) are
    reported under ``errors`` rather than aborting the whole extraction.
    """
    from hax.cli.cli import MODULES_DICT  # imported here so failures are localized

    programs: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, (module_path, description) in MODULES_DICT.items():
        try:
            programs[name] = schema_for_program(name, module_path, description)
        except Exception as exc:  # noqa: BLE001 - record and continue
            errors[name] = f"{type(exc).__name__}: {exc}"
    return {"programs": programs, "errors": errors}


if __name__ == "__main__":
    import json

    result = extract_all()
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
