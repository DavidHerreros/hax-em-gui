"""CRUD for the user's "/" chat commands (see :mod:`hax_gui.assistant_commands`)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import assistant_commands
from .shared import auth

router = APIRouter()


class CommandBody(BaseModel):
    description: str = ""
    prompt: str
    attach_run: bool = True
    include_tab: bool = False
    may_act: bool = False
    pinned: bool = False


@router.get("/api/assistant/commands", dependencies=[Depends(auth)])
def list_commands() -> list[dict[str, Any]]:
    return assistant_commands.list_commands()


@router.put("/api/assistant/commands/{name}", dependencies=[Depends(auth)])
def save_command(name: str, body: CommandBody) -> dict[str, Any]:
    try:
        return assistant_commands.save_command(name, **body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/api/assistant/commands/{name}", dependencies=[Depends(auth)])
def delete_command(name: str) -> dict[str, Any]:
    try:
        found = assistant_commands.delete_command(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not found:
        raise HTTPException(404, "no such command")
    return {"ok": True}
