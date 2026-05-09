from __future__ import annotations

import json

from agent_learning.config import AzureOpenAISettings
from agent_learning.models import AgentCard, BudgetState, LoopActionType, LoopDecision, UsageSnapshot
from agent_learning.usage import PricingCatalog


class DecisionPlanner:
    def __init__(self, settings: AzureOpenAISettings, pricing: PricingCatalog) -> None:
        self.settings = settings
        self.pricing = pricing

    def decide(
        self,
        *,
        agent: AgentCard,
        goal: str,
        context: str,
        budget: BudgetState,
        available_tools: list[str],
        available_subagents: list[str],
    ) -> tuple[LoopDecision, UsageSnapshot]:
        if self.settings.is_configured:
            decision, usage = self._decide_with_azure(
                agent=agent,
                goal=goal,
                context=context,
                budget=budget,
                available_tools=available_tools,
                available_subagents=available_subagents,
            )
            if decision is not None and usage is not None:
                return decision, usage

        return self._heuristic_decision(
            agent=agent,
            goal=goal,
            context=context,
            budget=budget,
            available_tools=available_tools,
            available_subagents=available_subagents,
        )

    def _decide_with_azure(
        self,
        *,
        agent: AgentCard,
        goal: str,
        context: str,
        budget: BudgetState,
        available_tools: list[str],
        available_subagents: list[str],
    ) -> tuple[LoopDecision | None, UsageSnapshot | None]:
        try:
            from openai import AzureOpenAI
        except ImportError:
            return None, None

        client = AzureOpenAI(
            api_key=self.settings.api_key,
            api_version=self.settings.api_version,
            azure_endpoint=self.settings.endpoint,
        )
        response = client.chat.completions.create(
            model=self.settings.deployment,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": self._build_system_prompt(agent, available_tools, available_subagents),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(goal, context, budget),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "loop_decision",
                    "strict": True,
                    "schema": LoopDecision.model_json_schema(),
                },
            },
        )

        content = response.choices[0].message.content or "{}"
        decision = LoopDecision.model_validate(json.loads(content))
        usage = self.pricing.make_snapshot(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
            completion_tokens=getattr(response.usage, "completion_tokens", 0),
            model_name=self.settings.deployment or self.pricing.model_name,
        )
        return decision, usage

    def _heuristic_decision(
        self,
        *,
        agent: AgentCard,
        goal: str,
        context: str,
        budget: BudgetState,
        available_tools: list[str],
        available_subagents: list[str],
    ) -> tuple[LoopDecision, UsageSnapshot]:
        lower_goal = goal.lower()
        lower_context = context.lower()
        tool_name: str | None = None
        delegate_to: str | None = None
        message = ""
        action_type = LoopActionType.FINISH
        thought = f"{agent.name} is using heuristic planning because Azure OpenAI is not configured."
        memory_note = f"{agent.name} inspected the goal and remaining budget ${budget.remaining_usd:.2f}."
        has_subagent_output = "subagent output" in lower_context
        has_tool_result = "tool " in lower_context and " returned:" in lower_context

        if budget.remaining_usd <= 0:
            message = "Budget exhausted before another step could run."
        elif has_subagent_output or has_tool_result:
            message = (
                f"{agent.name} completed a useful cycle and has enough context to stop. "
                "Review the latest memory entries to see the subagent or tool result."
            )
        elif "mcp" in lower_goal and "mcp_status" in available_tools:
            tool_name = "mcp_status"
            action_type = LoopActionType.TOOL
            message = "Inspect configured MCP servers."
        elif "time" in lower_goal and "clock" in available_tools:
            tool_name = "clock"
            action_type = LoopActionType.TOOL
            message = "Check the current UTC timestamp."
        elif "agent" in lower_goal and "list_agents" in available_tools:
            tool_name = "list_agents"
            action_type = LoopActionType.TOOL
            message = "Inspect the loaded markdown agent definitions."
        elif "memory" in lower_goal and "memory_snapshot" in available_tools:
            tool_name = "memory_snapshot"
            action_type = LoopActionType.TOOL
            message = "Inspect the current memory snapshot."
        elif (
            any(keyword in lower_goal for keyword in ("inspect", "research", "analyze", "explain"))
            and not has_subagent_output
            and available_subagents
            and agent.subagents
        ):
            delegate_to = agent.subagents[0]
            action_type = LoopActionType.DELEGATE
            message = f"Delegating to {delegate_to} for a focused first pass."
            thought = f"{agent.name} wants a narrower subagent pass before deciding how to finish."
        else:
            message = (
                f"{agent.name} is ready. Azure OpenAI is {'configured' if self.settings.is_configured else 'not configured'}, "
                "the loop skeleton is active, and the next slice can focus on deeper tool execution or UI polish."
            )

        if action_type is LoopActionType.TOOL:
            memory_note = f"{agent.name} chose tool {tool_name}."
        elif action_type is LoopActionType.DELEGATE:
            memory_note = f"{agent.name} delegated to subagent {delegate_to}."

        decision = LoopDecision(
            thought=thought,
            action_type=action_type,
            message=message,
            tool_name=tool_name,
            tool_input=goal if tool_name == "echo" else None,
            delegate_to=delegate_to,
            memory_note=memory_note,
        )
        usage = self.pricing.make_snapshot(
            prompt_tokens=max(len(goal.split()) * 2 + len(context.split()), 1),
            completion_tokens=max(len(message.split()), 1),
            model_name="heuristic-fallback",
        )
        return decision, usage

    @staticmethod
    def _build_system_prompt(
        agent: AgentCard,
        available_tools: list[str],
        available_subagents: list[str],
    ) -> str:
        return (
            f"You are the agent '{agent.name}'. {agent.description}\n\n"
            f"System prompt:\n{agent.system_prompt}\n\n"
            f"Tools: {', '.join(available_tools) or 'none'}\n"
            f"Subagents: {', '.join(available_subagents) or 'none'}\n"
            f"Skills: {', '.join(agent.skills) or 'none'}\n"
            f"Memory policy: {agent.memory_policy}\n"
            "Return a LoopDecision JSON object only."
        )

    @staticmethod
    def _build_user_prompt(goal: str, context: str, budget: BudgetState) -> str:
        return (
            f"Goal: {goal}\n\n"
            f"Context:\n{context or 'No prior context.'}\n\n"
            f"Budget: step=${budget.step_budget_usd:.2f}, run=${budget.run_budget_usd:.2f}, "
            f"spent=${budget.spent_usd:.4f}, remaining=${budget.remaining_usd:.4f}"
        )
