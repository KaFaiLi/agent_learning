from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_learning.agents import AgentRegistry
from agent_learning.config import RuntimeSettings
from agent_learning.hooks import HookBundle, HookManager
from agent_learning.llm import LLMClient, LLMReply
from agent_learning.models import (
    AgentCard,
    BudgetState,
    HookEvent,
    Message,
    RunReport,
    SkillCard,
    ToolCall,
    ToolResult,
    UsageSnapshot,
)
from agent_learning.skills import SkillRegistry
from agent_learning.store import JSONLStore
from agent_learning.tools import ToolContext, ToolRegistry
from agent_learning.tools.fs import register_fs_tools
from agent_learning.tools.introspect import register_introspect_tools
from agent_learning.tools.shell import register_shell_tools
from agent_learning.usage import PricingCatalog, merge_usage

EventListener = Callable[[dict[str, Any]], None]


@dataclass
class EngineDeps:
    settings: RuntimeSettings
    llm: LLMClient
    agents: AgentRegistry
    skills: SkillRegistry
    tools: ToolRegistry
    store: JSONLStore
    pricing: PricingCatalog
    mcp_bridge: Any | None = None
    listeners: list[EventListener] = field(default_factory=list)


def build_engine(settings: RuntimeSettings, *, llm: LLMClient, mcp_bridge: Any | None = None) -> "Engine":
    agents = AgentRegistry(settings.agent_dir)
    skills = SkillRegistry(settings.skill_dir)
    store = JSONLStore(settings.runtime_dir)
    tools = ToolRegistry()
    register_fs_tools(tools)
    register_shell_tools(tools)
    register_introspect_tools(tools)
    if mcp_bridge is not None:
        mcp_bridge.register_into(tools)
    pricing = PricingCatalog(settings.pricing)
    deps = EngineDeps(
        settings=settings,
        llm=llm,
        agents=agents,
        skills=skills,
        tools=tools,
        store=store,
        pricing=pricing,
        mcp_bridge=mcp_bridge,
    )
    return Engine(deps)


class Engine:
    def __init__(self, deps: EngineDeps) -> None:
        self.deps = deps

    # ---- public API -------------------------------------------------

    def add_listener(self, listener: EventListener) -> None:
        self.deps.listeners.append(listener)

    def snapshot(self) -> dict[str, Any]:
        return {
            "agents": [a.model_dump() for a in self.deps.agents.list()],
            "skills": [s.model_dump() for s in self.deps.skills.list()],
            "tools": self.deps.tools.names(),
            "azure_configured": self.deps.settings.azure.is_configured,
            "workspace": str(self.deps.settings.workspace),
            "mcp": self.deps.mcp_bridge.describe() if self.deps.mcp_bridge else "(no MCP)",
        }

    def refresh(self) -> None:
        self.deps.agents.refresh()
        self.deps.skills.refresh()

    def run_goal(self, goal: str, *, agent_name: str | None = None, budget: BudgetState | None = None) -> RunReport:
        self.refresh()
        budget = budget or BudgetState(
            step_budget_usd=self.deps.settings.step_budget_usd,
            run_budget_usd=self.deps.settings.run_budget_usd,
        )
        agent = self._resolve_agent(agent_name or self.deps.settings.default_agent)
        run_id = self.deps.store.start_run(goal)
        hooks = self._make_hooks(agent, run_id)
        self._emit({"kind": "run_started", "run_id": run_id, "goal": goal, "agent": agent.name})

        hooks.fire(HookEvent.SESSION_START, payload={"agent": agent.name, "goal": goal})
        prompt_response = hooks.fire(HookEvent.USER_PROMPT_SUBMIT, payload={"agent": agent.name, "goal": goal})
        if prompt_response.block:
            final = f"User prompt blocked by hook: {prompt_response.reason}"
            self._record_event(run_id, "system", final)
            self.deps.store.finish_run(run_id, final_message=final)
            hooks.fire(HookEvent.STOP, payload={"agent": agent.name, "final": final})
            return RunReport(run_id=run_id, goal=goal, final_message=final)

        extra_context = prompt_response.additional_context

        usages: list[UsageSnapshot] = []
        final, iterations = self._run_agent(
            run_id=run_id,
            agent=agent,
            goal=goal,
            extra_context=extra_context,
            budget=budget,
            hooks=hooks,
            usages=usages,
            depth=0,
        )

        self.deps.store.finish_run(run_id, final_message=final)
        hooks.fire(HookEvent.STOP, payload={"agent": agent.name, "final": final})
        self._emit({"kind": "run_finished", "run_id": run_id, "final": final})

        report = RunReport(
            run_id=run_id,
            goal=goal,
            final_message=final,
            iterations=iterations,
            usage=merge_usage(usages),
        )
        return report

    # ---- internals --------------------------------------------------

    def _resolve_agent(self, name: str) -> AgentCard:
        agent = self.deps.agents.get(name)
        if agent is not None:
            return agent
        agents = self.deps.agents.list()
        if agents:
            return agents[0]
        return AgentCard(
            name="goal-runner",
            description="Built-in fallback agent.",
            system_prompt="You are a careful coding assistant. Use available tools to make progress.",
            tools=["*"],
        )

    def _make_hooks(self, agent: AgentCard, run_id: str) -> HookManager:
        bundle = HookBundle.from_sources(
            settings_hooks=self.deps.settings.extra.get("hooks") if self.deps.settings.extra else None,
            agent_hooks=agent.hooks,
        )
        return HookManager(bundle, store=self.deps.store, run_id=run_id)

    def _build_messages(self, agent: AgentCard, goal: str, extra_context: str | None) -> list[Message]:
        skills = self.deps.skills.select(declared=agent.skills, goal=goal)
        skill_block = self.deps.skills.render(skills)
        tool_names = self._allowed_tool_names(agent)
        sys_parts = [
            f"You are the agent '{agent.name}'.",
            agent.description.strip(),
            "",
            agent.system_prompt.strip(),
            "",
            f"Workspace: {self.deps.settings.workspace}",
            f"Available tools: {', '.join(tool_names) or 'none'}",
        ]
        if agent.subagents:
            sys_parts.append(f"Sub-agents you can delegate to: {', '.join(agent.subagents)}")
        if skill_block:
            sys_parts.append("")
            sys_parts.append(skill_block)
        if extra_context:
            sys_parts.append("")
            sys_parts.append(f"Additional context from hooks:\n{extra_context}")
        sys_parts.append("")
        sys_parts.append(
            "Use tools as needed. When the goal is complete or you need user input, reply with plain text and no tool calls."
        )
        system = "\n".join(p for p in sys_parts if p is not None).strip()
        return [
            Message(role="system", content=system),
            Message(role="user", content=goal),
        ]

    def _allowed_tool_names(self, agent: AgentCard) -> list[str]:
        if not agent.tools or "*" in agent.tools:
            base = self.deps.tools.names()
        else:
            base = [n for n in self.deps.tools.names() if n in agent.tools]
        return base + list(agent.subagents)  # sub-agents are presented to the model as virtual tools

    def _tool_schemas(self, agent: AgentCard) -> list[dict[str, Any]]:
        if not agent.tools or "*" in agent.tools:
            allowed = None
        else:
            allowed = list(agent.tools)
        schemas = self.deps.tools.openai_schemas(allowed=allowed)
        for sub in agent.subagents:
            sub_card = self.deps.agents.get(sub)
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": sub,
                        "description": f"Delegate a task to sub-agent '{sub}'. {sub_card.description if sub_card else ''}".strip(),
                        "parameters": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "task": {"type": "string", "description": "Concrete task for the sub-agent."},
                                "context": {"type": "string", "description": "Optional extra context."},
                            },
                            "required": ["task", "context"],
                        },
                    },
                }
            )
        return schemas

    def _run_agent(
        self,
        *,
        run_id: str,
        agent: AgentCard,
        goal: str,
        extra_context: str | None,
        budget: BudgetState,
        hooks: HookManager,
        usages: list[UsageSnapshot],
        depth: int,
    ) -> tuple[str, int]:
        hooks.fire(HookEvent.AGENT_START, payload={"agent": agent.name, "goal": goal, "depth": depth})
        self._emit({"kind": "agent_start", "agent": agent.name, "goal": goal, "depth": depth})

        messages = self._build_messages(agent, goal, extra_context)
        ctx = self._tool_context(run_id, agent)
        tool_schemas = self._tool_schemas(agent)
        max_iter = max(1, agent.max_iterations if depth == 0 else max(2, agent.max_iterations // 2))
        final_text = ""
        iterations = 0

        for iteration in range(1, max_iter + 1):
            iterations = iteration
            if budget.remaining_usd <= 0:
                final_text = "Budget exhausted."
                break

            self._emit({"kind": "iteration", "agent": agent.name, "i": iteration, "depth": depth})
            try:
                reply = self.deps.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as exc:
                final_text = f"LLM call failed: {type(exc).__name__}: {exc}"
                self._record_event(run_id, agent.name, final_text)
                break

            usage = self.deps.pricing.snapshot(
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                model=reply.model,
            )
            usages.append(usage)
            budget.spent_usd += usage.estimated_cost_usd
            self.deps.store.record(run_id, kind="usage", role=agent.name, payload=usage.model_dump())

            messages.append(reply.message)
            if reply.message.content:
                self._record_event(run_id, agent.name, reply.message.content)
                self._emit({"kind": "assistant", "agent": agent.name, "content": reply.message.content, "depth": depth})

            if not reply.message.tool_calls:
                final_text = reply.message.content or ""
                break

            for tc in reply.message.tool_calls:
                result = self._dispatch_tool(
                    run_id=run_id,
                    agent=agent,
                    call=tc,
                    ctx=ctx,
                    hooks=hooks,
                    budget=budget,
                    usages=usages,
                    depth=depth,
                )
                messages.append(
                    Message(role="tool", tool_call_id=tc.id, name=tc.name, content=result.content)
                )

        else:
            final_text = final_text or "Iteration limit reached."

        hooks.fire(HookEvent.AGENT_STOP, payload={"agent": agent.name, "final": final_text, "depth": depth})
        self._emit({"kind": "agent_stop", "agent": agent.name, "final": final_text, "depth": depth})
        return final_text or "(no output)", iterations

    def _dispatch_tool(
        self,
        *,
        run_id: str,
        agent: AgentCard,
        call: ToolCall,
        ctx: ToolContext,
        hooks: HookManager,
        budget: BudgetState,
        usages: list[UsageSnapshot],
        depth: int,
    ) -> ToolResult:
        self._record_event(run_id, agent.name, f"tool_call: {call.name}({_short(call.arguments)})", kind="tool_call")
        self._emit({"kind": "tool_call", "agent": agent.name, "tool": call.name, "arguments": call.arguments, "depth": depth})

        pre = hooks.fire(HookEvent.PRE_TOOL_USE, matcher_value=call.name, payload={"tool": call.name, "arguments": call.arguments, "agent": agent.name})
        if pre.block:
            result = ToolResult(tool_call_id=call.id, name=call.name, content=f"blocked: {pre.reason}", ok=False)
            self._record_tool_result(run_id, agent, result, depth)
            return result
        if pre.transform:
            call = call.model_copy(update={"arguments": {**call.arguments, **pre.transform}})

        if call.name in agent.subagents:
            result = self._run_subagent_as_tool(
                run_id=run_id,
                parent=agent,
                sub_name=call.name,
                call=call,
                budget=budget,
                hooks=hooks,
                usages=usages,
                depth=depth,
            )
        else:
            result = self.deps.tools.execute(call, ctx)

        post = hooks.fire(HookEvent.POST_TOOL_USE, matcher_value=call.name, payload={"tool": call.name, "result": result.content, "ok": result.ok})
        if post.transform and "content" in post.transform:
            result = result.model_copy(update={"content": str(post.transform["content"])})
        self._record_tool_result(run_id, agent, result, depth)
        return result

    def _run_subagent_as_tool(
        self,
        *,
        run_id: str,
        parent: AgentCard,
        sub_name: str,
        call: ToolCall,
        budget: BudgetState,
        hooks: HookManager,
        usages: list[UsageSnapshot],
        depth: int,
    ) -> ToolResult:
        if depth >= 1:
            return ToolResult(tool_call_id=call.id, name=call.name, content="Sub-agent depth exceeded.", ok=False)
        sub = self.deps.agents.get(sub_name)
        if sub is None:
            return ToolResult(tool_call_id=call.id, name=call.name, content=f"Unknown sub-agent: {sub_name}", ok=False)
        sub_goal = call.arguments.get("task") or "(no task)"
        sub_extra = call.arguments.get("context")
        text, _ = self._run_agent(
            run_id=run_id,
            agent=sub,
            goal=sub_goal,
            extra_context=sub_extra,
            budget=budget,
            hooks=hooks,
            usages=usages,
            depth=depth + 1,
        )
        return ToolResult(tool_call_id=call.id, name=call.name, content=text or "(no output)", ok=True)

    def _tool_context(self, run_id: str, agent: AgentCard) -> ToolContext:
        return ToolContext(
            workspace=self.deps.settings.workspace,
            agent_registry=self.deps.agents,
            skill_registry=self.deps.skills,
            mcp_bridge=self.deps.mcp_bridge,
            store=self.deps.store,
            run_id=run_id,
            bash_allow=list(agent.bash_allow),
            bash_deny=list(agent.bash_deny),
            todos=[],
        )

    def _record_event(self, run_id: str, role: str, message: str, *, kind: str = "event") -> None:
        self.deps.store.record(run_id, kind=kind, role=role, payload={"message": message})

    def _record_tool_result(self, run_id: str, agent: AgentCard, result: ToolResult, depth: int) -> None:
        self.deps.store.record(
            run_id,
            kind="tool_result",
            role=result.name,
            payload={"ok": result.ok, "content": result.content[:8000]},
        )
        self._emit({"kind": "tool_result", "agent": agent.name, "tool": result.name, "ok": result.ok, "content": result.content, "depth": depth})

    def _emit(self, event: dict[str, Any]) -> None:
        for listener in list(self.deps.listeners):
            try:
                listener(event)
            except Exception:
                pass


def _short(value: Any, *, limit: int = 160) -> str:
    s = str(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."
