from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import os
import sys
import unittest
from unittest.mock import patch

from agent_learning.cli import main
from agent_learning.config import AzureOpenAISettings, PricingSettings, RuntimeSettings
from agent_learning.models import AgentCard, RunReport, UsageSnapshot


class RuntimeSettingsTests(unittest.TestCase):
    def test_runtime_settings_from_env_applies_overrides(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
                "AZURE_OPENAI_API_KEY": "secret",
                "AZURE_OPENAI_DEPLOYMENT": "gpt-demo",
                "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
                "AGENT_LEARNING_MODEL_NAME": "priced-model",
                "AGENT_LEARNING_INPUT_PRICE_PER_1K_TOKENS": "1.25",
                "AGENT_LEARNING_OUTPUT_PRICE_PER_1K_TOKENS": "2.5",
                "AGENT_LEARNING_AGENT_DIR": str(root / "custom_agents"),
                "AGENT_LEARNING_SKILL_DIR": str(root / "custom_skills"),
                "AGENT_LEARNING_MCP_CONFIG": str(root / "custom_mcp.yaml"),
                "AGENT_LEARNING_MEMORY_DB": str(root / "state" / "runtime.db"),
                "AGENT_LEARNING_DEFAULT_GOAL": "inspect env config",
                "AGENT_LEARNING_MAX_ITERATIONS": "9",
                "AGENT_LEARNING_STEP_BUDGET_USD": "0.75",
                "AGENT_LEARNING_RUN_BUDGET_USD": "7.5",
            }
            with patch.dict(os.environ, env, clear=False):
                settings = RuntimeSettings.from_env(root)

            self.assertEqual(settings.root_dir, root)
            self.assertTrue(settings.azure.is_configured)
            self.assertEqual(settings.azure.api_version, "2025-01-01-preview")
            self.assertEqual(settings.pricing.model_name, "priced-model")
            self.assertEqual(settings.pricing.input_cost_per_1k_tokens, 1.25)
            self.assertEqual(settings.pricing.output_cost_per_1k_tokens, 2.5)
            self.assertEqual(settings.agent_dir, root / "custom_agents")
            self.assertEqual(settings.skill_dir, root / "custom_skills")
            self.assertEqual(settings.mcp_config_path, root / "custom_mcp.yaml")
            self.assertEqual(settings.memory_db_path, root / "state" / "runtime.db")
            self.assertEqual(settings.default_goal, "inspect env config")
            self.assertEqual(settings.max_iterations, 9)
            self.assertEqual(settings.default_step_budget_usd, 0.75)
            self.assertEqual(settings.default_run_budget_usd, 7.5)

    def test_azure_settings_defaults_to_unconfigured(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = AzureOpenAISettings.from_env()

        self.assertFalse(settings.is_configured)
        self.assertEqual(settings.api_version, "2024-10-21")

    def test_ensure_storage_dirs_creates_memory_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = RuntimeSettings(
                root_dir=root,
                azure=AzureOpenAISettings(endpoint=None, api_key=None, deployment=None),
                pricing=PricingSettings(
                    model_name="demo",
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                agent_dir=root / "agents",
                skill_dir=root / "skills",
                mcp_config_path=root / "mcp_servers.yaml",
                memory_db_path=root / "state" / "runtime.db",
                default_goal="inspect",
                max_iterations=3,
                default_step_budget_usd=0.5,
                default_run_budget_usd=5.0,
            )

            settings.ensure_storage_dirs()

            self.assertTrue((root / "state").is_dir())


class CLITests(unittest.TestCase):
    def test_main_print_config_outputs_resolved_settings(self) -> None:
        with TemporaryDirectory() as tmp:
            stdout = StringIO()
            with patch.object(sys, "argv", ["main.py", "--print-config", "--root", tmp]):
                with patch("agent_learning.cli.RuntimeEngine") as engine_cls:
                    with redirect_stdout(stdout):
                        main()

            engine_cls.assert_called_once()
            output = stdout.getvalue()
            self.assertIn(f"agent_dir={Path(tmp) / 'demo_agents'}", output)
            self.assertIn(f"mcp_config_path={Path(tmp) / 'mcp_servers.yaml'}", output)
            self.assertIn(f"memory_db_path={Path(tmp) / '.agent_learning' / 'runtime.db'}", output)
            self.assertIn("azure_configured=False", output)
            self.assertIn("pricing_model_name=azure-deployment", output)

    def test_main_once_runs_engine_and_prints_summary(self) -> None:
        report = RunReport(
            run_id="run-123",
            goal="inspect once",
            final_message="done",
            agents=[AgentCard(name="goal-runner", description="Main agent")],
            memories=[],
            usage=UsageSnapshot(
                model="heuristic-fallback",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                estimated_cost_usd=0.01,
                tool_calls=1,
            ),
        )
        stdout = StringIO()
        with patch.object(sys, "argv", ["main.py", "--once", "--goal", "inspect once"]):
            with patch("agent_learning.cli.RuntimeEngine") as engine_cls:
                engine_cls.return_value.run_goal.return_value = report
                with redirect_stdout(stdout):
                    main()

        engine_cls.return_value.run_goal.assert_called_once_with("inspect once")
        output = stdout.getvalue()
        self.assertIn("Run run-123: done", output)
        self.assertIn("Agents: goal-runner", output)
        self.assertIn("Memories: 0", output)
        self.assertIn("Usage: model=heuristic-fallback total_tokens=15 estimated_cost_usd=0.0100 tool_calls=1", output)

    def test_main_launches_tui_with_requested_goal(self) -> None:
        fake_module = ModuleType("agent_learning.tui.app")

        class FakeApp:
            instance: "FakeApp | None" = None

            def __init__(self, *, engine: object, initial_goal: str) -> None:
                self.engine = engine
                self.initial_goal = initial_goal
                self.run_called = False
                FakeApp.instance = self

            def run(self) -> None:
                self.run_called = True

        fake_module.AgentLearningApp = FakeApp

        with patch.object(sys, "argv", ["main.py", "--goal", "inspect ui"]):
            with patch("agent_learning.cli.RuntimeEngine") as engine_cls:
                with patch.dict(sys.modules, {"agent_learning.tui.app": fake_module}):
                    main()

        self.assertIsNotNone(FakeApp.instance)
        assert FakeApp.instance is not None
        self.assertIs(FakeApp.instance.engine, engine_cls.return_value)
        self.assertEqual(FakeApp.instance.initial_goal, "inspect ui")
        self.assertTrue(FakeApp.instance.run_called)


if __name__ == "__main__":
    unittest.main()
