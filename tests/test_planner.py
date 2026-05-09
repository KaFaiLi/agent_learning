from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import json
import sys
import unittest
from unittest.mock import patch

from agent_learning.config import AzureOpenAISettings, PricingSettings
from agent_learning.llm import DecisionPlanner
from agent_learning.models import AgentCard, BudgetState, LoopActionType
from agent_learning.usage import PricingCatalog


class DecisionPlannerTests(unittest.TestCase):
    def _planner(self, *, azure: AzureOpenAISettings | None = None) -> DecisionPlanner:
        return DecisionPlanner(
            azure or AzureOpenAISettings(endpoint=None, api_key=None, deployment=None),
            PricingCatalog.from_settings(
                PricingSettings(
                    model_name="demo-model",
                    input_cost_per_1k_tokens=0.25,
                    output_cost_per_1k_tokens=0.5,
                )
            ),
        )

    def test_decide_uses_tool_branch_for_time_goal(self) -> None:
        decision, usage = self._planner().decide(
            agent=AgentCard(name="goal-runner", description="Main agent"),
            goal="tell me the time",
            context="Recent memory:\nNo memory yet.",
            budget=BudgetState(),
            available_tools=["clock", "echo"],
            available_subagents=[],
        )

        self.assertEqual(decision.action_type, LoopActionType.TOOL)
        self.assertEqual(decision.tool_name, "clock")
        self.assertEqual(decision.message, "Check the current UTC timestamp.")
        self.assertEqual(decision.memory_note, "goal-runner chose tool clock.")
        self.assertEqual(usage.model, "heuristic-fallback")
        self.assertGreater(usage.total_tokens, 0)

    def test_decide_delegates_when_analysis_goal_has_subagent(self) -> None:
        decision, _ = self._planner().decide(
            agent=AgentCard(
                name="goal-runner",
                description="Main agent",
                subagents=["researcher"],
            ),
            goal="analyze the runtime state",
            context="Recent memory:\nNo memory yet.",
            budget=BudgetState(),
            available_tools=["clock"],
            available_subagents=["researcher"],
        )

        self.assertEqual(decision.action_type, LoopActionType.DELEGATE)
        self.assertEqual(decision.delegate_to, "researcher")
        self.assertIn("Delegating to researcher", decision.message)

    def test_decide_finishes_after_tool_result_is_present_in_context(self) -> None:
        decision, _ = self._planner().decide(
            agent=AgentCard(name="goal-runner", description="Main agent"),
            goal="tell me the time",
            context="[working] Tool clock returned: 2026-05-09T00:00:00+00:00",
            budget=BudgetState(),
            available_tools=["clock"],
            available_subagents=[],
        )

        self.assertEqual(decision.action_type, LoopActionType.FINISH)
        self.assertIn("completed a useful cycle", decision.message)

    def test_decide_stops_when_budget_is_exhausted(self) -> None:
        decision, _ = self._planner().decide(
            agent=AgentCard(name="goal-runner", description="Main agent"),
            goal="inspect memory",
            context="No memory yet.",
            budget=BudgetState(run_budget_usd=0.0, spent_usd=0.0),
            available_tools=["memory_snapshot"],
            available_subagents=[],
        )

        self.assertEqual(decision.action_type, LoopActionType.FINISH)
        self.assertEqual(decision.message, "Budget exhausted before another step could run.")

    def test_prompt_builders_include_runtime_details(self) -> None:
        agent = AgentCard(
            name="goal-runner",
            description="Main agent",
            system_prompt="Plan carefully.",
            skills=["inspect"],
        )

        system_prompt = DecisionPlanner._build_system_prompt(
            agent,
            available_tools=["clock", "echo"],
            available_subagents=["researcher"],
        )
        user_prompt = DecisionPlanner._build_user_prompt(
            "inspect state",
            "Recent memory",
            BudgetState(step_budget_usd=0.5, run_budget_usd=5.0, spent_usd=1.25),
        )

        self.assertIn("You are the agent 'goal-runner'.", system_prompt)
        self.assertIn("Tools: clock, echo", system_prompt)
        self.assertIn("Subagents: researcher", system_prompt)
        self.assertIn("Skills: inspect", system_prompt)
        self.assertIn("Goal: inspect state", user_prompt)
        self.assertIn("spent=$1.2500", user_prompt)
        self.assertIn("remaining=$3.7500", user_prompt)

    def test_decide_uses_azure_response_when_configuration_is_present(self) -> None:
        planner = self._planner(
            azure=AzureOpenAISettings(
                endpoint="https://example.openai.azure.com",
                api_key="secret",
                deployment="gpt-deploy",
            )
        )

        @dataclass
        class FakeUsage:
            prompt_tokens: int
            completion_tokens: int

        class FakeAzureOpenAI:
            calls: list[dict[str, object]] = []

            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self.create)
                )

            def create(self, **kwargs: object) -> object:
                FakeAzureOpenAI.calls.append(kwargs)
                payload = {
                    "thought": "Use Azure result",
                    "action_type": "finish",
                    "message": "Azure planned the next step.",
                    "tool_name": None,
                    "tool_input": None,
                    "delegate_to": None,
                    "memory_note": "Persist Azure summary.",
                }
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
                    usage=FakeUsage(prompt_tokens=12, completion_tokens=8),
                )

        fake_openai = SimpleNamespace(AzureOpenAI=FakeAzureOpenAI)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            decision, usage = planner.decide(
                agent=AgentCard(name="goal-runner", description="Main agent"),
                goal="inspect azure path",
                context="No memory yet.",
                budget=BudgetState(),
                available_tools=["clock"],
                available_subagents=[],
            )

        self.assertEqual(decision.action_type, LoopActionType.FINISH)
        self.assertEqual(decision.message, "Azure planned the next step.")
        self.assertEqual(decision.memory_note, "Persist Azure summary.")
        self.assertEqual(usage.model, "gpt-deploy")
        self.assertEqual(usage.prompt_tokens, 12)
        self.assertEqual(usage.completion_tokens, 8)
        self.assertEqual(usage.total_tokens, 20)
        self.assertEqual(len(FakeAzureOpenAI.calls), 1)
        self.assertEqual(FakeAzureOpenAI.calls[0]["model"], "gpt-deploy")


if __name__ == "__main__":
    unittest.main()
