from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sqlite3
import uuid

from agent_learning.models import MemoryEntry, MemoryKind, RunEvent, UsageSnapshot


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS run_events (
                run_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS usage_events (
                run_id TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                tool_calls INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def start_run(self, goal: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.connection.execute(
            "INSERT INTO runs (id, goal, status, started_at) VALUES (?, ?, ?, ?)",
            (run_id, goal, "running", self._now()),
        )
        self.connection.commit()
        return run_id

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        self.connection.execute(
            "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
            (status, self._now(), run_id),
        )
        self.connection.commit()

    def record_event(self, run_id: str, role: str, message: str) -> None:
        self.connection.execute(
            "INSERT INTO run_events (run_id, role, message, created_at) VALUES (?, ?, ?, ?)",
            (run_id, role, message, self._now()),
        )
        self.connection.commit()

    def append_memory(self, run_id: str, kind: MemoryKind, content: str) -> None:
        self.connection.execute(
            "INSERT INTO memories (run_id, kind, content, created_at) VALUES (?, ?, ?, ?)",
            (run_id, kind.value, content, self._now()),
        )
        self.connection.commit()

    def record_usage(self, run_id: str, usage: UsageSnapshot) -> None:
        self.connection.execute(
            """
            INSERT INTO usage_events (
                run_id,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                estimated_cost_usd,
                tool_calls,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                usage.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                usage.estimated_cost_usd,
                usage.tool_calls,
                self._now(),
            ),
        )
        self.connection.commit()

    def load_memories(self, run_id: str) -> list[MemoryEntry]:
        rows = self.connection.execute(
            "SELECT kind, content, created_at FROM memories WHERE run_id = ? ORDER BY rowid ASC",
            (run_id,),
        ).fetchall()
        return [
            MemoryEntry(
                kind=MemoryKind(row["kind"]),
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def load_events(self, run_id: str) -> list[RunEvent]:
        rows = self.connection.execute(
            "SELECT role, message, created_at FROM run_events WHERE run_id = ? ORDER BY rowid ASC",
            (run_id,),
        ).fetchall()
        return [
            RunEvent(
                role=row["role"],
                message=row["message"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def load_usage_events(self, run_id: str) -> list[UsageSnapshot]:
        rows = self.connection.execute(
            """
            SELECT model, prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd, tool_calls
            FROM usage_events
            WHERE run_id = ?
            ORDER BY rowid ASC
            """,
            (run_id,),
        ).fetchall()
        return [
            UsageSnapshot(
                model=row["model"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
                estimated_cost_usd=row["estimated_cost_usd"],
                tool_calls=row["tool_calls"],
            )
            for row in rows
        ]

    def recent_memory_text(self, run_id: str, limit: int = 6) -> str:
        rows = self.connection.execute(
            "SELECT kind, content FROM memories WHERE run_id = ? ORDER BY rowid DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
        ordered = list(reversed(rows))
        return "\n".join(f"[{row['kind']}] {row['content']}" for row in ordered)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
