from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from agent_learning.models import ToolCall, ToolResult


@dataclass
class ToolContext:
    workspace: Path
    agent_registry: Any | None = None
    skill_registry: Any | None = None
    mcp_bridge: Any | None = None
    store: Any | None = None
    run_id: str | None = None
    bash_allow: list[str] = field(default_factory=list)
    bash_deny: list[str] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: str
    description: str
    args_model: type[BaseModel]

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult: ...


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    return _patch_strict(schema)


def _patch_strict(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        if "required" not in schema:
            schema["required"] = list(schema["properties"].keys())
    for key in ("properties", "$defs"):
        if key in schema:
            schema[key] = {k: _patch_strict(v) for k, v in schema[key].items()}
    for key in ("anyOf", "allOf", "oneOf"):
        if key in schema:
            schema[key] = [_patch_strict(v) for v in schema[key]]
    if "items" in schema:
        schema["items"] = _patch_strict(schema["items"])
    return schema


@dataclass
class _ToolEntry:
    tool: Tool
    factory: Callable[[BaseModel, ToolContext], ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = _ToolEntry(tool=tool, factory=tool.run)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        entry = self._tools.get(name)
        return entry.tool if entry else None

    def openai_schemas(self, *, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, entry in sorted(self._tools.items()):
            if allowed is not None and "*" not in allowed and name not in allowed:
                continue
            schema = _strict_schema(entry.tool.args_model)
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": entry.tool.description,
                        "parameters": schema,
                    },
                }
            )
        return out

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        entry = self._tools.get(call.name)
        if entry is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Unknown tool: {call.name}",
                ok=False,
            )
        try:
            args = entry.tool.args_model.model_validate(call.arguments or {})
        except ValidationError as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Invalid arguments: {exc}",
                ok=False,
            )
        try:
            result = entry.factory(args, ctx)
        except Exception as exc:  # tools should not crash the loop
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Tool error: {type(exc).__name__}: {exc}",
                ok=False,
            )
        if not result.tool_call_id:
            result = result.model_copy(update={"tool_call_id": call.id})
        return result


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
