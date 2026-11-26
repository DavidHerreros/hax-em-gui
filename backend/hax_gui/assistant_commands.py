"""User-defined "/" commands for the assistant chat.

Each command is one small text file, ``<data_dir>/assistant_commands/<name>.md``:
a ``key: value`` header between ``---`` lines, then the prompt. Plain files so a
command can be copied to a colleague. The chat expands the prompt in the browser;
nothing here talks to the model, so this works without the assistant package.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import settings

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
# Handled by the chat itself: a file with one of these names would never run.
RESERVED = frozenset({"autonomy", "attach", "report", "float", "new", "commands"})
FLAGS = {"attach_run": True, "include_tab": False, "may_act": False, "pinned": False}
MAX_PROMPT = 8000


def commands_dir() -> Path:
    return settings.data_dir / "assistant_commands"


def check_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise ValueError("a command name is 1-32 lowercase letters, digits or dashes, not starting with a dash")
    if name in RESERVED:
        raise ValueError(f"/{name} is a built-in command")
    return name


def _parse(name: str, text: str) -> dict[str, Any]:
    cmd: dict[str, Any] = {"name": name, "description": "", "prompt": text.strip(), **FLAGS}
    m = re.match(r"\A---\n(.*?)\n---\n?(.*)\Z", text, re.S)
    if not m:
        return cmd
    cmd["prompt"] = m.group(2).strip()
    for line in m.group(1).splitlines():
        key, sep, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not sep:
            continue
        if key == "description":
            cmd["description"] = value
        elif key in FLAGS:
            cmd[key] = value.lower() in ("true", "yes", "1")
    return cmd


def _dump(cmd: dict[str, Any]) -> str:
    head = [f"description: {cmd['description']}"] + [f"{k}: {'true' if cmd[k] else 'false'}" for k in FLAGS]
    return "---\n" + "\n".join(head) + "\n---\n" + cmd["prompt"] + "\n"


def list_commands() -> list[dict[str, Any]]:
    d = commands_dir()
    if not d.is_dir():
        return []
    return [_parse(p.stem, p.read_text(encoding="utf-8"))
            for p in sorted(d.glob("*.md")) if NAME_RE.match(p.stem) and p.stem not in RESERVED]


def save_command(name: str, description: str, prompt: str, **flags: Any) -> dict[str, Any]:
    check_name(name)
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("the command needs a prompt")
    if len(prompt) > MAX_PROMPT:
        raise ValueError(f"the prompt is longer than {MAX_PROMPT} characters")
    cmd = {"name": name, "description": " ".join((description or "").split())[:200], "prompt": prompt,
           **{k: bool(flags.get(k, default)) for k, default in FLAGS.items()}}
    d = commands_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(_dump(cmd), encoding="utf-8")
    return cmd


def delete_command(name: str) -> bool:
    check_name(name)
    path = commands_dir() / f"{name}.md"
    if not path.is_file():
        return False
    path.unlink()
    return True
