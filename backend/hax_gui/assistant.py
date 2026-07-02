"""Optional bridge to the ``hax-em-assistant`` package.

The GUI must run whether or not the assistant is installed, so everything here is
guarded: :func:`available` reports installation, :func:`status` returns a plain
dict (``{"available": False}`` when absent), and the chat streamer is only reached
after an availability check. Nothing at import time requires the assistant.
"""

from __future__ import annotations

from typing import Iterator, Optional

try:  # the assistant is an independent, optional package
    from hax_assistant.session import AssistantSession
    _INSTALLED = True
except Exception:  # noqa: BLE001 - ImportError, or a broken/incompatible install
    AssistantSession = None  # type: ignore[assignment,misc]
    _INSTALLED = False

_session: Optional["AssistantSession"] = None


def available() -> bool:
    return _INSTALLED


def _get() -> "AssistantSession":
    global _session
    if _session is None:
        _session = AssistantSession()  # type: ignore[operator]
    return _session


def status() -> dict:
    """Availability + (when installed) model/grounding info for the SPA."""
    if not _INSTALLED:
        return {"available": False}
    try:
        return {"available": True, **_get().status()}
    except Exception as e:  # noqa: BLE001 - surface a bad config without crashing the GUI
        return {"available": False, "error": str(e)}


def stream_reply(messages: list[dict], grounded: bool = True,
                 run: Optional[dict] = None) -> Iterator[str]:
    """Yield reply tokens for a conversation (list of {role, content})."""
    return _get().stream_reply(messages, grounded=grounded, run=run)


def stream_events(messages: list[dict], grounded: bool = True,
                  run: Optional[dict] = None) -> Iterator[dict]:
    """Yield typed events (status activity + content tokens) for a conversation.

    ``run`` (optional) is a dict of resolved artifact paths for an attached run:
    ``{"run_dir", "latents", "volumes", "label"}`` — the session loads those
    deterministically and injects a features summary for the model to narrate.
    """
    return _get().stream_events(messages, grounded=grounded, run=run)
