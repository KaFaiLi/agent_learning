from __future__ import annotations

from pydantic import BaseModel, Field

from agent_learning.models import ToolResult
from agent_learning.tools import ToolContext


def _ok(name: str, content: str) -> ToolResult:
    return ToolResult(tool_call_id="", name=name, content=content, ok=True)


class _Empty(BaseModel):
    pass


class ListAgentsTool:
    name = "list_agents"
    description = "List available markdown agents."
    args_model = _Empty

    def run(self, args: _Empty, ctx: ToolContext) -> ToolResult:
        registry = ctx.agent_registry
        if registry is None:
            return _ok(self.name, "(no agent registry)")
        cards = registry.list()
        if not cards:
            return _ok(self.name, "(no agents loaded)")
        return _ok(self.name, "\n".join(f"- {c.name}: {c.description}" for c in cards))


class ListSkillsTool:
    name = "list_skills"
    description = "List available markdown skills with their triggers."
    args_model = _Empty

    def run(self, args: _Empty, ctx: ToolContext) -> ToolResult:
        registry = ctx.skill_registry
        if registry is None:
            return _ok(self.name, "(no skill registry)")
        cards = registry.list()
        if not cards:
            return _ok(self.name, "(no skills loaded)")
        lines = []
        for c in cards:
            triggers = ", ".join(c.triggers) if c.triggers else "(none)"
            lines.append(f"- {c.name}: {c.description} [triggers: {triggers}]")
        return _ok(self.name, "\n".join(lines))


class MCPStatusTool:
    name = "mcp_status"
    description = "Show configured MCP servers and connection state."
    args_model = _Empty

    def run(self, args: _Empty, ctx: ToolContext) -> ToolResult:
        bridge = ctx.mcp_bridge
        if bridge is None:
            return _ok(self.name, "(MCP bridge unavailable)")
        return _ok(self.name, bridge.describe())


class MemorySnapshotTool:
    name = "memory_snapshot"
    description = "Recent run events, useful when you need to recall what's already happened."
    args_model = _Empty

    def run(self, args: _Empty, ctx: ToolContext) -> ToolResult:
        if ctx.store is None or ctx.run_id is None:
            return _ok(self.name, "(no run in progress)")
        return _ok(self.name, ctx.store.recent_text(ctx.run_id) or "(empty)")


class TodoWriteArgs(BaseModel):
    todos: list[dict] = Field(description="List of {content, status, activeForm} entries.")


class TodoWriteTool:
    name = "todo_write"
    description = "Replace the agent's TODO list. Each entry: {content, status: pending|in_progress|completed, activeForm}."
    args_model = TodoWriteArgs

    def run(self, args: TodoWriteArgs, ctx: ToolContext) -> ToolResult:
        ctx.todos.clear()
        ctx.todos.extend(args.todos)
        return _ok(self.name, f"Recorded {len(args.todos)} todo(s).")


class TodoReadTool:
    name = "todo_read"
    description = "Read the agent's current TODO list."
    args_model = _Empty

    def run(self, args: _Empty, ctx: ToolContext) -> ToolResult:
        if not ctx.todos:
            return _ok(self.name, "(no todos)")
        lines = [f"- [{t.get('status', '?')}] {t.get('content', '')}" for t in ctx.todos]
        return _ok(self.name, "\n".join(lines))


def register_introspect_tools(registry) -> None:
    for tool in (
        ListAgentsTool(),
        ListSkillsTool(),
        MCPStatusTool(),
        MemorySnapshotTool(),
        TodoWriteTool(),
        TodoReadTool(),
    ):
        registry.register(tool)
