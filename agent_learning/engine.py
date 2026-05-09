from __future__ import annotations

from agent_learning.agent_defs import AgentRegistry
from agent_learning.config import RuntimeSettings
from agent_learning.hooks import HookManager
from agent_learning.llm import DecisionPlanner
from agent_learning.mcp import MCPBridge
from agent_learning.memory import MemoryStore
from agent_learning.models import AgentCard, BudgetState, LoopActionType, MemoryKind, RunEvent, RunReport
from agent_learning.skills import SkillRegistry
from agent_learning.tools import ToolContext, ToolRegistry
from agent_learning.usage import PricingCatalog, merge_usage


class RuntimeEngine:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.store = MemoryStore(settings.memory_store_path)
        self.agents = AgentRegistry(settings.agent_dir)
        self.skills = SkillRegistry(settings.skill_dir)
        self.mcp_bridge = MCPBridge(settings.mcp_config_path)
        self.pricing = PricingCatalog.from_settings(settings.pricing)
        self.planner = DecisionPlanner(settings.azure, self.pricing)
        self.hooks = HookManager(self.store)

    def refresh(self) -> None:
        self.agents.refresh()
        self.skills.refresh()
        self.mcp_bridge.refresh()

    def run_goal(self, goal: str) -> RunReport:
        self.refresh()
        run_id = self.store.start_run(goal)
        budget = BudgetState(
            step_budget_usd=self.settings.default_step_budget_usd,
            run_budget_usd=self.settings.default_run_budget_usd,
        )
        self.store.record_event(run_id, "system", f"Started run for goal: {goal}")
        self.store.append_memory(run_id, MemoryKind.WORKING, f"Goal: {goal}")

        final_message = "Run did not reach a finish state."

        for iteration in range(1, self.settings.max_iterations + 1):
            agent = self._root_agent()
            decision = self._run_agent_step(run_id, goal, agent, budget, iteration, allow_delegate=True)

            if decision["done"]:
                final_message = decision["message"]
                break

        else:
            final_message = "Iteration limit reached before the loop declared success."

        self.store.append_memory(run_id, MemoryKind.SUMMARY, final_message)
        self.store.record_event(run_id, "system", final_message)
        self.store.finish_run(run_id)
        return self._build_report(run_id, goal, final_message)

    def _run_agent_step(
        self,
        run_id: str,
        goal: str,
        agent: AgentCard,
        budget: BudgetState,
        iteration: int,
        *,
        allow_delegate: bool,
    ) -> dict[str, str | bool]:
        context = self._build_context(run_id, budget, iteration)
        tools = ToolRegistry(
            ToolContext(
                agent_registry=self.agents,
                skill_registry=self.skills,
                mcp_bridge=self.mcp_bridge,
                memory_snapshot=lambda: self.store.recent_memory_text(run_id),
            )
        )
        self.hooks.fire(run_id, agent, "before_plan", details=f"Iteration {iteration}")

        available_subagents = agent.subagents if allow_delegate else []
        decision, usage = self.planner.decide(
            agent=agent,
            goal=goal,
            context=context,
            budget=budget,
            available_tools=tools.names(),
            available_subagents=available_subagents,
        )
        budget.spent_usd += usage.estimated_cost_usd
        self.store.record_usage(run_id, usage)
        self.store.record_event(
            run_id,
            agent.name,
            f"Iteration {iteration}: {decision.thought}",
        )

        if decision.memory_note:
            self.store.append_memory(run_id, MemoryKind.WORKING, decision.memory_note)

        if budget.remaining_usd <= 0:
            self.hooks.fire(run_id, agent, "after_finish", details="Budget exhausted.")
            return {"done": True, "message": "The run stopped because the budget was exhausted."}

        if decision.action_type is LoopActionType.TOOL and decision.tool_name:
            tool_result = tools.execute(decision.tool_name, decision.tool_input or "")
            self.store.record_event(
                run_id,
                decision.tool_name,
                tool_result.output,
            )
            self.store.append_memory(
                run_id,
                MemoryKind.WORKING,
                f"Tool {decision.tool_name} returned: {tool_result.output}",
            )
            self.store.record_usage(
                run_id,
                usage.model_copy(update={"tool_calls": 1, "estimated_cost_usd": 0.0}),
            )
            self.hooks.fire(
                run_id,
                agent,
                "after_tool",
                details=f"{decision.tool_name}: {tool_result.output}",
            )
            return {"done": False, "message": tool_result.output}

        if decision.action_type is LoopActionType.DELEGATE and decision.delegate_to:
            subagent = self.agents.get(decision.delegate_to)
            if subagent is None:
                missing_message = f"Requested subagent '{decision.delegate_to}' was not found."
                self.store.record_event(run_id, "system", missing_message)
                return {"done": True, "message": missing_message}

            delegated = self._run_subagent(run_id, goal, subagent, budget, iteration)
            self.store.append_memory(
                run_id,
                MemoryKind.WORKING,
                f"Subagent output: {delegated}",
            )
            self.hooks.fire(run_id, agent, "after_delegate", details=delegated)
            return {"done": False, "message": delegated}

        if decision.action_type is LoopActionType.FINISH:
            self.hooks.fire(run_id, agent, "after_finish", details=decision.message)
            return {"done": True, "message": decision.message}

        return {"done": False, "message": decision.message}

    def _run_subagent(
        self,
        run_id: str,
        goal: str,
        agent: AgentCard,
        budget: BudgetState,
        iteration: int,
    ) -> str:
        context = self._build_context(run_id, budget, iteration)
        tools = ToolRegistry(
            ToolContext(
                agent_registry=self.agents,
                skill_registry=self.skills,
                mcp_bridge=self.mcp_bridge,
                memory_snapshot=lambda: self.store.recent_memory_text(run_id),
            )
        )
        self.hooks.fire(run_id, agent, "before_plan", details=f"Subagent iteration {iteration}")
        decision, usage = self.planner.decide(
            agent=agent,
            goal=goal,
            context=context,
            budget=budget,
            available_tools=tools.names(),
            available_subagents=[],
        )
        budget.spent_usd += usage.estimated_cost_usd
        self.store.record_usage(run_id, usage)
        self.store.record_event(run_id, agent.name, f"Subagent thought: {decision.thought}")

        if decision.action_type is LoopActionType.TOOL and decision.tool_name:
            tool_result = tools.execute(decision.tool_name, decision.tool_input or "")
            self.store.record_event(run_id, decision.tool_name, tool_result.output)
            self.hooks.fire(
                run_id,
                agent,
                "after_tool",
                details=f"{decision.tool_name}: {tool_result.output}",
            )
            return f"{agent.name} used {decision.tool_name}: {tool_result.output}"

        self.hooks.fire(run_id, agent, "after_finish", details=decision.message or decision.thought)
        return decision.message or decision.thought

    def _root_agent(self) -> AgentCard:
        root = self.agents.get("goal-runner")
        if root is not None:
            return root

        agents = self.agents.list_agents()
        if agents:
            return agents[0]

        return AgentCard(
            name="goal-runner",
            description="Fallback goal runner when no agent markdown files are present.",
            system_prompt="Run a simple observe-plan-act loop with memory and budget tracking.",
            tools=["echo", "clock", "list_agents", "list_skills", "memory_snapshot", "mcp_status"],
            subagents=[],
        )

    def _build_context(self, run_id: str, budget: BudgetState, iteration: int) -> str:
        recent_memory = self.store.recent_memory_text(run_id)
        return (
            f"Iteration: {iteration}\n"
            f"Budget spent: ${budget.spent_usd:.4f}\n"
            f"Remaining budget: ${budget.remaining_usd:.4f}\n"
            f"Recent memory:\n{recent_memory or 'No memory yet.'}"
        )

    def _build_report(self, run_id: str, goal: str, final_message: str) -> RunReport:
        return RunReport(
            run_id=run_id,
            goal=goal,
            final_message=final_message,
            agents=self.agents.list_agents(),
            skills=self.skills.list_skills(),
            memories=self.store.load_memories(run_id),
            events=self.store.load_events(run_id),
            usage=merge_usage(self.store.load_usage_events(run_id)),
        )

    def snapshot(self) -> dict[str, object]:
        self.refresh()
        return {
            "agents": self.agents.list_agents(),
            "skills": self.skills.list_skills(),
            "mcp": self.mcp_bridge.list_servers(),
            "azure_configured": self.settings.azure.is_configured,
        }

