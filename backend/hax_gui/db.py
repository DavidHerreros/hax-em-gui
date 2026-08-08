"""Shared, thread-safe SQLite access for the GUI's managers.

Four managers (jobs, labels, workflows, chats) store their tables in the same
database file. Each used to hold its *own* ``sqlite3.connect(...,
check_same_thread=False)`` and use it concurrently from the event loop and the
request threadpool — which is unsupported (one connection, many threads, no
lock), and, across connections, contended the default rollback journal: a
colliding writer blocked for the 5 s default and then raised ``database is
locked`` as a bare 500.

This module fixes both by construction:

- **One connection per database file, shared by every manager** — a single
  writer, so cross-connection journal contention cannot happen at all.
- **A reentrant lock around every statement**, taken for the *materialised*
  result: ``execute`` prefetches the rows inside the lock and hands back a
  cursor stand-in, so no caller ever fetches from a live cursor another thread
  could disturb.
- **WAL + busy_timeout + synchronous=NORMAL** so readers never block the
  writer, and any residual lock wait fails loudly only after a real timeout.

``transaction()`` exposes the lock as a context manager for multi-statement
read-modify-write blocks (rename-vs-persist races, paired DELETEs): held across
the block, no other thread's statements or commits can interleave.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Iterator


class PrefetchedCursor:
    """The result of one statement, detached from the live cursor.

    Rows are fetched eagerly inside the connection lock; this object then
    serves ``fetchone``/``fetchall``/iteration from memory, plus the
    ``rowcount`` the statement reported. Result sets here are small (job
    lists, label sets, workflow graphs), so eager fetch costs nothing.
    """

    def __init__(self, rows: list, rowcount: int, lastrowid: int | None) -> None:
        self._rows = rows
        self._i = 0
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchone(self) -> Any | None:
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        return row

    def fetchall(self) -> list:
        rows = self._rows[self._i:]
        self._i = len(self._rows)
        return rows

    def __iter__(self) -> Iterator:
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row


class LockedConnection:
    """A serialised ``sqlite3.Connection`` stand-in (execute/commit subset)."""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA synchronous=NORMAL")

    def execute(self, sql: str, params: tuple = ()) -> PrefetchedCursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            return PrefetchedCursor(rows, cur.rowcount, cur.lastrowid)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def transaction(self) -> threading.RLock:
        """Hold the connection for a whole read-modify-write block.

        Reentrant, so ``execute``/``commit`` inside the block just work."""
        return self._lock


_connections: dict[str, LockedConnection] = {}
_registry_lock = threading.Lock()


def connect(path) -> LockedConnection:
    """The shared connection for ``path`` (created on first use)."""
    key = str(path)
    with _registry_lock:
        conn = _connections.get(key)
        if conn is None:
            conn = _connections[key] = LockedConnection(key)
        return conn
