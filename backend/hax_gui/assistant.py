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
                  run: Optional[dict] = None, allow_actions: bool = False) -> Iterator[dict]:
    """Yield typed events (status activity + content tokens + action proposals).

    ``run`` (optional) is a dict of resolved artifact paths for an attached run:
    ``{"run_dir", "latents", "volumes", "label"}`` — the session loads those
    deterministically and injects a features summary for the model to narrate.
    ``allow_actions`` lets the session emit deterministic action proposals (create
    a draft job from a data path); the GUI confirms before anything happens.
    """
    return _get().stream_events(messages, grounded=grounded, run=run, allow_actions=allow_actions)


def generate_report(out_path: str, *, run_dir: Optional[str] = None,
                    latents: Optional[str] = None, volumes: Optional[str] = None,
                    title: str = "hax-em run report", subtitle: str = "",
                    narrate: bool = True) -> dict:
    """Build a deterministic run-report PDF at ``out_path`` (figures + measured
    values), optionally with an AI-generated discussion section. Returns the
    report manifest (included/skipped sections). Requires the assistant package."""
    from hax_assistant.report import ReportInputs, build_report

    session = _get()
    measured, _steps = session.build_run_context(run_dir=run_dir, latents=latents, volumes=volumes)
    # Request only the discussion sections whose data is present, so each
    # paragraph lines up with the figure it introduces in the report.
    keys = ["overview"]
    if latents:
        keys.append("landscape")
    if run_dir:
        keys.append("metrics")
    if volumes:
        keys.append("motion")
    sections = (session.narrate_run_sections(measured, keys)
                if (narrate and measured) else {})
    inp = ReportInputs(title=title, subtitle=subtitle, run_dir=run_dir, latents=latents,
                       volumes=volumes, measured_text=measured, sections=sections)
    return build_report(out_path, inp)
