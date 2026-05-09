from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_learning.config import AzureOpenAISettings, PricingSettings, RuntimeSettings
from agent_learning.engine import RuntimeEngine


REPO_ROOT = Path(__file__).resolve().parents[1]


class RuntimeEngineTests(unittest.TestCase):
    def _make_settings(self, root: Path) -> RuntimeSettings:
        return RuntimeSettings(
            root_dir=root,
            azure=AzureOpenAISettings(endpoint=None, api_key=None, deployment=None),
            pricing=PricingSettings(
                model_name="heuristic-fallback",
                input_cost_per_1k_tokens=0.0,
                output_cost_per_1k_tokens=0.0,
            ),
            agent_dir=REPO_ROOT / "demo_agents",
            skill_dir=REPO_ROOT / "demo_skills",
            mcp_config_path=root / "mcp_servers.yaml",
            memory_db_path=root / ".agent_learning" / "runtime.db",
            default_goal="inspect runtime",
            max_iterations=3,
            default_step_budget_usd=0.5,
            default_run_budget_usd=5.0,
        )

    def test_run_goal_delegates_to_subagent_and_finishes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._make_settings(root)
            settings.ensure_storage_dirs()
            engine = RuntimeEngine(settings)

            report = engine.run_goal("inspect the current runtime state")

            self.assertIn("completed a useful cycle", report.final_message)
            self.assertEqual(report.goal, "inspect the current runtime state")
            self.assertEqual(report.agents[0].name, "builder")
            self.assertTrue(any(memory.content.startswith("Subagent output:") for memory in report.memories))
            self.assertTrue(any(event.role == "hook:after_delegate" for event in report.events))
            self.assertGreater(report.usage.total_tokens, 0)
            engine.store.connection.close()

    def test_run_goal_executes_tool_and_records_tool_usage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._make_settings(root)
            settings.ensure_storage_dirs()
            engine = RuntimeEngine(settings)

            report = engine.run_goal("tell me the time right now")

            self.assertIn("completed a useful cycle", report.final_message)
            self.assertTrue(any(event.role == "clock" for event in report.events))
            self.assertTrue(any("Tool clock returned:" in memory.content for memory in report.memories))
            self.assertEqual(report.usage.tool_calls, 1)
            engine.store.connection.close()


if __name__ == "__main__":
    unittest.main()
