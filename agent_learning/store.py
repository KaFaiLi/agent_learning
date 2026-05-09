from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_learning.models import RunEvent


class JSONLStore:
    """Append-only per-run JSONL event log."""

    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir
        self.runs_dir = runtime_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.jsonl"

    def start_run(self, goal: str) -> str:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.record(run_id, kind="run_started", role="system", payload={"goal": goal})
        return run_id

    def finish_run(self, run_id: str, *, final_message: str) -> None:
        self.record(run_id, kind="run_finished", role="system", payload={"final_message": final_message})

    def record(self, run_id: str, *, kind: str, role: str = "system", payload: dict[str, Any] | None = None) -> RunEvent:
        event = RunEvent(run_id=run_id, kind=kind, role=role, payload=payload or {})
        line = event.model_dump_json() + "\n"
        with self._lock:
            with self._path(run_id).open("a", encoding="utf-8") as fh:
                fh.write(line)
        return event

    def load(self, run_id: str) -> list[RunEvent]:
        path = self._path(run_id)
        if not path.exists():
            return []
        events: list[RunEvent] = []
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    events.append(RunEvent.model_validate(json.loads(raw)))
                except Exception:  # tolerate corrupt lines
                    continue
        return events

    def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self.runs_dir.glob("*.jsonl"))

    def recent_text(self, run_id: str, *, limit: int = 8) -> str:
        events = [e for e in self.load(run_id) if e.kind in ("event", "memory", "tool_result")]
        recent = events[-limit:]
        if not recent:
            return ""
        return "\n".join(f"[{e.role}] {self._render(e)}" for e in recent)

    @staticmethod
    def _render(event: RunEvent) -> str:
        payload = event.payload
        if "message" in payload:
            return str(payload["message"])
        if "content" in payload:
            return str(payload["content"])
        return json.dumps(payload, ensure_ascii=False)
