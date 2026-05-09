from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from agent_learning.agent_defs import AgentRegistry
from agent_learning.mcp import MCPBridge
from agent_learning.models import ToolResult
from agent_learning.skills import SkillRegistry


@dataclass(slots=True)
class ToolContext:
    agent_registry: AgentRegistry
    skill_registry: SkillRegistry
    mcp_bridge: MCPBridge
    memory_snapshot: Callable[[], str]


class ToolRegistry:
    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self._handlers: dict[str, Callable[[str], ToolResult]] = {
            "echo": self._echo,
            "clock": self._clock,
            "list_agents": self._list_agents,
            "list_skills": self._list_skills,
            "memory_snapshot": self._memory_snapshot,
            "mcp_status": self._mcp_status,
        }

    def names(self) -> list[str]:
        return sorted(self._handlers.keys())

    def execute(self, name: str, payload: str = "") -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(tool_name=name, output=f"Unknown tool: {name}", ok=False)
        return handler(payload)

    def _echo(self, payload: str) -> ToolResult:
        return ToolResult(tool_name="echo", output=payload or "(empty input)")

    def _clock(self, _: str) -> ToolResult:
        return ToolResult(tool_name="clock", output=datetime.now(UTC).isoformat())

    def _list_agents(self, _: str) -> ToolResult:
        agents = self.context.agent_registry.list_agents()
        if not agents:
            return ToolResult(tool_name="list_agents", output="No agent markdown files found.")
        output = "\n".join(f"- {agent.name}: {agent.description}" for agent in agents)
        return ToolResult(tool_name="list_agents", output=output)

    def _list_skills(self, _: str) -> ToolResult:
        skills = self.context.skill_registry.list_skills()
        if not skills:
            return ToolResult(tool_name="list_skills", output="No skill markdown files found.")
        output = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)
        return ToolResult(tool_name="list_skills", output=output)

    def _memory_snapshot(self, _: str) -> ToolResult:
        snapshot = self.context.memory_snapshot()
        output = snapshot or "No memory entries recorded yet."
        return ToolResult(tool_name="memory_snapshot", output=output)

    def _mcp_status(self, _: str) -> ToolResult:
        return ToolResult(tool_name="mcp_status", output=self.context.mcp_bridge.describe())
