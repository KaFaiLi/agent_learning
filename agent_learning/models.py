from __future__ import annotations

from datetime import datetime, UTC
from enum import Enum
from pydantic import BaseModel, Field


class MemoryKind(str, Enum):
    WORKING = "working"
    SUMMARY = "summary"
    PERSISTED = "persisted"


class LoopActionType(str, Enum):
    THINK = "think"
    TOOL = "tool"
    DELEGATE = "delegate"
    FINISH = "finish"


class BudgetState(BaseModel):
    step_budget_usd: float = Field(default=0.50, ge=0)
    run_budget_usd: float = Field(default=5.00, ge=0)
    spent_usd: float = Field(default=0.0, ge=0)

    @property
    def remaining_usd(self) -> float:
        return max(self.run_budget_usd - self.spent_usd, 0.0)


class UsageSnapshot(BaseModel):
    model: str = "unconfigured"
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    tool_calls: int = Field(default=0, ge=0)


class MemoryEntry(BaseModel):
    kind: MemoryKind
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentCard(BaseModel):
    name: str
    description: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    subagents: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    hooks: dict[str, list[str]] = Field(default_factory=dict)
    memory_policy: str = "Summarize important changes and decisions between steps."


class LoopDecision(BaseModel):
    thought: str
    action_type: LoopActionType
    message: str = ""
    tool_name: str | None = None
    tool_input: str | None = None
    delegate_to: str | None = None
    memory_note: str = ""


class ToolResult(BaseModel):
    tool_name: str
    output: str
    ok: bool = True


class RunEvent(BaseModel):
    role: str
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SkillCard(BaseModel):
    name: str
    description: str
    instructions: str = ""


class RunReport(BaseModel):
    run_id: str
    goal: str
    final_message: str
    agents: list[AgentCard] = Field(default_factory=list)
    skills: list[SkillCard] = Field(default_factory=list)
    memories: list[MemoryEntry] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)
    usage: UsageSnapshot = Field(default_factory=UsageSnapshot)