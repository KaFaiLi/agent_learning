from __future__ import annotations

from agent_learning.memory import MemoryStore
from agent_learning.models import AgentCard, MemoryKind


class HookManager:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def fire(self, run_id: str, agent: AgentCard, event_name: str, *, details: str = "") -> None:
        for template in agent.hooks.get(event_name, []):
            message = template.format(agent=agent.name, details=details)
            self.store.record_event(run_id, f"hook:{event_name}", message)
            if event_name.startswith("after_"):
                self.store.append_memory(run_id, MemoryKind.WORKING, message)
