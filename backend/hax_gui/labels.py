"""Job labels: a small shared palette of named, colored tags attached to jobs.

Labels are global (not per-project) so the same tag can group/filter runs across
projects. A job may carry several labels (many-to-many via ``job_labels``). Stored
in the same SQLite database as jobs/workflows.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from . import db
from .config import settings

# A pleasant default palette offered when creating a label (the UI may override).
DEFAULT_COLORS = [
    "#ef4444", "#f59e0b", "#eab308", "#22c55e", "#14b8a6",
    "#3b82f6", "#8b5cf6", "#ec4899", "#64748b", "#0ea5e9",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LabelManager:
    def __init__(self) -> None:
        self._db: db.LockedConnection | None = None

    def init(self) -> None:
        settings.ensure_dirs()
        self._db = db.connect(settings.db_path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                color TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS job_labels (
                job_id TEXT NOT NULL,
                label_id TEXT NOT NULL,
                PRIMARY KEY (job_id, label_id)
            )
            """
        )
        self._db.commit()

    # -- label CRUD --------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        rows = self._db.execute(
            """
            SELECT l.*, (SELECT COUNT(*) FROM job_labels jl WHERE jl.label_id = l.id) AS count
            FROM labels l ORDER BY l.name COLLATE NOCASE
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get(self, lid: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM labels WHERE id=?", (lid,)).fetchone()
        return dict(row) if row else None

    def create(self, name: str, color: str | None) -> dict[str, Any]:
        lid = uuid.uuid4().hex[:12]
        name = (name or "").strip() or "label"
        color = (color or "").strip() or DEFAULT_COLORS[hash(name) % len(DEFAULT_COLORS)]
        self._db.execute(
            "INSERT INTO labels (id, name, color, created_at) VALUES (?,?,?,?)",
            (lid, name, color, _now()),
        )
        self._db.commit()
        return self.get(lid)

    def update(self, lid: str, name: str | None = None, color: str | None = None) -> dict[str, Any]:
        if self.get(lid) is None:
            raise KeyError(lid)
        sets, params = [], []
        if name is not None:
            sets.append("name=?"); params.append(name.strip() or "label")
        if color is not None and color.strip():
            sets.append("color=?"); params.append(color.strip())
        if sets:
            params.append(lid)
            self._db.execute(f"UPDATE labels SET {', '.join(sets)} WHERE id=?", params)
            self._db.commit()
        return self.get(lid)

    def delete(self, lid: str) -> None:
        with self._db.transaction():   # both DELETEs land in one commit
            self._db.execute("DELETE FROM job_labels WHERE label_id=?", (lid,))
            self._db.execute("DELETE FROM labels WHERE id=?", (lid,))
            self._db.commit()

    def remove_jobs(self, job_ids: list[str]) -> None:
        """Drop the associations of deleted jobs (their labels themselves stay).
        Without this every deleted job left its job_labels rows behind forever."""
        if not job_ids:
            return
        marks = ",".join("?" for _ in job_ids)
        self._db.execute(f"DELETE FROM job_labels WHERE job_id IN ({marks})", tuple(job_ids))
        self._db.commit()

    # -- job <-> label associations ---------------------------------------
    def set_job_labels(self, job_id: str, label_ids: list[str]) -> list[dict[str, Any]]:
        """Replace a job's labels with the given set (unknown ids are ignored)."""
        valid = {r["id"] for r in self._db.execute("SELECT id FROM labels").fetchall()}
        self._db.execute("DELETE FROM job_labels WHERE job_id=?", (job_id,))
        for lid in dict.fromkeys(label_ids):       # de-dupe, keep order
            if lid in valid:
                self._db.execute("INSERT OR IGNORE INTO job_labels (job_id, label_id) VALUES (?,?)", (job_id, lid))
        self._db.commit()
        return self.labels_for(job_id)

    def labels_for(self, job_id: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            """
            SELECT l.* FROM labels l JOIN job_labels jl ON jl.label_id = l.id
            WHERE jl.job_id=? ORDER BY l.name COLLATE NOCASE
            """,
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def labels_for_many(self, job_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """{job_id: [labels]} for a batch of jobs (one query)."""
        out: dict[str, list[dict[str, Any]]] = {jid: [] for jid in job_ids}
        if not job_ids:
            return out
        marks = ",".join("?" for _ in job_ids)
        rows = self._db.execute(
            f"""
            SELECT jl.job_id AS job_id, l.* FROM job_labels jl
            JOIN labels l ON l.id = jl.label_id
            WHERE jl.job_id IN ({marks}) ORDER BY l.name COLLATE NOCASE
            """,
            tuple(job_ids),
        ).fetchall()
        for r in rows:
            d = dict(r)
            jid = d.pop("job_id")
            out.setdefault(jid, []).append(d)
        return out


manager = LabelManager()
