"""Assistant chat history: persisted conversations, optionally grouped by project.

Each chat stores its full message list (a JSON blob) plus a title and an optional
``project_id`` so the UI can group chats under their project. Stored in the same
SQLite database as jobs/workflows/labels. This is plain data and exists whether or
not the optional assistant package is installed.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatManager:
    def __init__(self) -> None:
        self._db: sqlite3.Connection | None = None

    def init(self) -> None:
        settings.ensure_dirs()
        self._db = sqlite3.connect(settings.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
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
        row = self._db.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
        if not row:
            return None
        new_title = title.strip() if isinstance(title, str) and title.strip() else row["title"]
        new_pid = project_id if project_id_set else row["project_id"]
        new_msgs = json.dumps(messages) if messages is not None else row["messages"]
        self._db.execute(
            "UPDATE chats SET title=?, project_id=?, messages=?, updated_at=? WHERE id=?",
            (new_title, new_pid, new_msgs, _now(), cid),
        )
        self._db.commit()
        return self.get(cid)

    def delete(self, cid: str) -> bool:
        cur = self._db.execute("DELETE FROM chats WHERE id=?", (cid,))
        self._db.commit()
        return cur.rowcount > 0


manager = ChatManager()
