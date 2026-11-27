"""Assistant chat history: persisted conversations, optionally grouped by project.

Each chat stores its full message list (a JSON blob) plus a title and an optional
``project_id`` so the UI can group chats under their project. Stored in the same
SQLite database as jobs/workflows/labels. This is plain data and exists whether or
not the optional assistant package is installed.

The server may post messages into a chat (the run monitor's findings): they carry
``posted`` and an ``id``, and ``after`` (how many of the chat's own messages came
before). The client saves the whole list, so a save keeps the posted messages it has
not seen yet, at their place, and the server's version of each (only ``seen`` is the
client's).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from . import db
from .config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge(client: list, stored: list) -> list:
    """The client's list with the server's posted messages: its copies updated, the unseen ones placed."""
    posted = {m["id"]: m for m in stored if isinstance(m, dict) and m.get("posted") and m.get("id")}
    out = []
    for m in client:
        if isinstance(m, dict) and m.get("posted") and m.get("id") in posted:
            m = {**posted.pop(m["id"]), "seen": bool(m.get("seen"))}
        out.append(m)
    for m in posted.values():
        own, at = 0, len(out)
        for i, x in enumerate(out):
            if own == m.get("after", 0):
                at = i
                break
            own += not (isinstance(x, dict) and x.get("posted"))
        while at < len(out) and isinstance(out[at], dict) and out[at].get("posted"):
            at += 1                       # after the posted messages already there
        out.insert(at, m)
    return out


class ChatManager:
    def __init__(self) -> None:
        self._db: db.LockedConnection | None = None

    def init(self) -> None:
        settings.ensure_dirs()
        self._db = db.connect(settings.db_path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                title TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._db.commit()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _meta(row: sqlite3.Row) -> dict[str, Any]:
        msgs = json.loads(row["messages"] or "[]")
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "title": row["title"],
            "updated_at": row["updated_at"],
            "created_at": row["created_at"],
            "message_count": len(msgs),
            "unread": any(isinstance(m, dict) and m.get("posted") and not m.get("seen") for m in msgs),
        }

    # -- CRUD --------------------------------------------------------------
    def list(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        if project_id is not None:
            rows = self._db.execute(
                "SELECT * FROM chats WHERE project_id=? ORDER BY updated_at DESC", (project_id,)
            ).fetchall()
        else:
            rows = self._db.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()
        return [self._meta(r) for r in rows]

    def get(self, cid: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
        if not row:
            return None
        d = self._meta(row)
        d["messages"] = json.loads(row["messages"] or "[]")
        return d

    def create(self, project_id: Optional[str], title: Optional[str]) -> dict[str, Any]:
        cid = uuid.uuid4().hex[:12]
        now = _now()
        title = (title or "").strip() or "New chat"
        self._db.execute(
            "INSERT INTO chats (id, project_id, title, messages, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (cid, project_id or None, title, "[]", now, now),
        )
        self._db.commit()
        return self.get(cid)

    def update(self, cid: str, *, title: Optional[str] = None,
               project_id: Optional[str] = None, project_id_set: bool = False,
               messages: Optional[list] = None) -> dict[str, Any] | None:
        # Read-modify-write under the connection lock: a rename racing the
        # end-of-turn message persist used to overwrite whichever landed first
        # with the other's stale snapshot (lost update).
        with self._db.transaction():
            row = self._db.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
            if not row:
                return None
            new_title = title.strip() if isinstance(title, str) and title.strip() else row["title"]
            new_pid = project_id if project_id_set else row["project_id"]
            new_msgs = (json.dumps(_merge(messages, json.loads(row["messages"] or "[]")))
                        if messages is not None else row["messages"])
            self._db.execute(
                "UPDATE chats SET title=?, project_id=?, messages=?, updated_at=? WHERE id=?",
                (new_title, new_pid, new_msgs, _now(), cid),
            )
            self._db.commit()
        return self.get(cid)

    def post(self, cid: str, message: dict[str, Any]) -> dict[str, Any] | None:
        """Append a server message (``posted``, ``id``) after the chat's current messages."""
        with self._db.transaction():
            row = self._db.execute("SELECT messages FROM chats WHERE id=?", (cid,)).fetchone()
            if not row:
                return None
            msgs = json.loads(row["messages"] or "[]")
            own = sum(1 for m in msgs if not (isinstance(m, dict) and m.get("posted")))
            message = {**message, "after": own, "seen": False}
            self._db.execute("UPDATE chats SET messages=?, updated_at=? WHERE id=?",
                             (json.dumps(msgs + [message]), _now(), cid))
            self._db.commit()
        return message

    def patch_posted(self, cid: str, mid: str, **fields: Any) -> dict[str, Any] | None:
        """Update fields of a posted message (e.g. its finding was decided)."""
        with self._db.transaction():
            row = self._db.execute("SELECT messages FROM chats WHERE id=?", (cid,)).fetchone()
            if not row:
                return None
            msgs = json.loads(row["messages"] or "[]")
            hit = next((m for m in msgs if isinstance(m, dict) and m.get("posted") and m.get("id") == mid), None)
            if hit is None:
                return None
            hit.update(fields)
            self._db.execute("UPDATE chats SET messages=?, updated_at=? WHERE id=?",
                             (json.dumps(msgs), _now(), cid))
            self._db.commit()
        return hit

    def delete(self, cid: str) -> bool:
        cur = self._db.execute("DELETE FROM chats WHERE id=?", (cid,))
        self._db.commit()
        return cur.rowcount > 0

    def delete_for_project(self, project_id: str) -> None:
        """Drop a deleted project's chats; they used to linger, grouped under a
        project id that no longer resolves."""
        self._db.execute("DELETE FROM chats WHERE project_id=?", (project_id,))
        self._db.commit()


manager = ChatManager()
