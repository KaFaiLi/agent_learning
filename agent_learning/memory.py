from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import threading
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from agent_learning.models import MemoryEntry, MemoryKind, RunEvent, UsageSnapshot


class MemoryStore:
    _SCHEMA = pa.schema(
        [
            ("sequence", pa.int64()),
            ("record_type", pa.string()),
            ("run_id", pa.string()),
            ("created_at", pa.string()),
            ("goal", pa.string()),
            ("status", pa.string()),
            ("role", pa.string()),
            ("message", pa.string()),
            ("memory_kind", pa.string()),
            ("content", pa.string()),
            ("model", pa.string()),
            ("prompt_tokens", pa.int64()),
            ("completion_tokens", pa.int64()),
            ("total_tokens", pa.int64()),
            ("estimated_cost_usd", pa.float64()),
            ("tool_calls", pa.int64()),
        ]
    )

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._lock = threading.RLock()
        self.initialize()

    def initialize(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not self.storage_path.exists():
                self._write_records([])

    def start_run(self, goal: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        self._append_record(
            "run_started",
            run_id,
            goal=goal,
            status="running",
        )
        return run_id

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        self._append_record("run_finished", run_id, status=status)

    def record_event(self, run_id: str, role: str, message: str) -> None:
        self._append_record("event", run_id, role=role, message=message)

    def append_memory(self, run_id: str, kind: MemoryKind, content: str) -> None:
        self._append_record("memory", run_id, memory_kind=kind.value, content=content)

    def record_usage(self, run_id: str, usage: UsageSnapshot) -> None:
        self._append_record(
            "usage",
            run_id,
            model=usage.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            tool_calls=usage.tool_calls,
        )

    def load_memories(self, run_id: str) -> list[MemoryEntry]:
        rows = self._records_for(run_id, "memory")
        return [
            MemoryEntry(
                kind=MemoryKind(row["memory_kind"]),
                content=row["content"] or "",
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def load_events(self, run_id: str) -> list[RunEvent]:
        rows = self._records_for(run_id, "event")
        return [
            RunEvent(
                role=row["role"] or "system",
                message=row["message"] or "",
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def load_usage_events(self, run_id: str) -> list[UsageSnapshot]:
        rows = self._records_for(run_id, "usage")
        return [
            UsageSnapshot(
                model=row["model"] or "unconfigured",
                prompt_tokens=row["prompt_tokens"] or 0,
                completion_tokens=row["completion_tokens"] or 0,
                total_tokens=row["total_tokens"] or 0,
                estimated_cost_usd=row["estimated_cost_usd"] or 0.0,
                tool_calls=row["tool_calls"] or 0,
            )
            for row in rows
        ]

    def recent_memory_text(self, run_id: str, limit: int = 6) -> str:
        rows = self._records_for(run_id, "memory")[-limit:]
        return "\n".join(
            f"[{row['memory_kind']}] {row['content'] or ''}" for row in rows
        )

    def _append_record(
        self,
        record_type: str,
        run_id: str,
        **fields: object,
    ) -> None:
        with self._lock:
            records = self._read_records()
            record = self._empty_record()
            record["sequence"] = len(records)
            record["record_type"] = record_type
            record["run_id"] = run_id
            record["created_at"] = self._now()
            for key, value in fields.items():
                record[key] = value
            records.append(record)
            self._write_records(records)

    def _records_for(self, run_id: str, record_type: str) -> list[dict[str, object]]:
        with self._lock:
            records = self._read_records()
        return [
            record
            for record in records
            if record["run_id"] == run_id and record["record_type"] == record_type
        ]

    def _read_records(self) -> list[dict[str, object]]:
        if not self.storage_path.exists():
            return []
        table = pq.read_table(self.storage_path)
        rows = table.to_pylist()
        return sorted(rows, key=lambda row: int(row["sequence"]))

    def _write_records(self, records: list[dict[str, object]]) -> None:
        table = pa.Table.from_pylist(records, schema=self._SCHEMA)
        pq.write_table(table, self.storage_path)

    @classmethod
    def _empty_record(cls) -> dict[str, object | None]:
        return {name: None for name in cls._SCHEMA.names}

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
