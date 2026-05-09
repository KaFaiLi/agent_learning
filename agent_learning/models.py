from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class HookEvent(StrEnum):
    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    AGENT_START = "AgentStart"
    AGENT_STOP = "AgentStop"
    STOP = "Stop"


class HookSpec(BaseModel):
    event: HookEvent
    matcher: str | None = None
    type: Literal["template", "command"] = "template"
    run: str
    timeout_s: int = 10


class HookResponse(BaseModel):
    block: bool = False
    reason: str | None = None
    transform: dict[str, Any] | None = None
    additional_context: str | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    content: str
    ok: bool = True


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _json_dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        return msg


class AcceptanceSpec(BaseModel):
    rubric: str | None = None
    verify_command: str | None = None
    max_outer_iterations: int = 5


class AgentCard(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=lambda: ["*"])
    subagents: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    hooks: dict[HookEvent, list[HookSpec]] = Field(default_factory=dict)
    bash_allow: list[str] = Field(default_factory=list)
    bash_deny: list[str] = Field(default_factory=list)
    acceptance: AcceptanceSpec | None = None
    max_iterations: int = 12


class SkillCard(BaseModel):
    name: str
    description: str = ""
    triggers: list[str] = Field(default_factory=list)
    body: str = ""


class BudgetState(BaseModel):
    step_budget_usd: float = 0.5
    run_budget_usd: float = 5.0
    spent_usd: float = 0.0

    @property
    def remaining_usd(self) -> float:
        return max(self.run_budget_usd - self.spent_usd, 0.0)


class UsageSnapshot(BaseModel):
    model: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    tool_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class EvalVerdict(BaseModel):
    met: bool
    gaps: str = ""
    rationale: str = ""


class RunEvent(BaseModel):
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    run_id: str
    kind: str
    role: str = "system"
    payload: dict[str, Any] = Field(default_factory=dict)


class RunReport(BaseModel):
    run_id: str
    goal: str
    final_message: str
    iterations: int = 0
    usage: UsageSnapshot = Field(default_factory=UsageSnapshot)
    verdict: EvalVerdict | None = None


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
