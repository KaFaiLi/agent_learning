from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_learning.agent_defs import AgentRegistry
from agent_learning.hooks import HookManager
from agent_learning.mcp import MCPBridge
from agent_learning.memory import MemoryStore
from agent_learning.models import AgentCard, MemoryKind, UsageSnapshot
from agent_learning.skills import SkillRegistry
from agent_learning.tools import ToolContext, ToolRegistry
from agent_learning.usage import PricingCatalog, merge_usage
from agent_learning.config import PricingSettings


class UsageTests(unittest.TestCase):
    def test_pricing_catalog_creates_and_merges_usage_snapshots(self) -> None:
        catalog = PricingCatalog.from_settings(
            PricingSettings(
                model_name="demo-model",
                input_cost_per_1k_tokens=0.25,
                output_cost_per_1k_tokens=0.5,
            )
        )

        first = catalog.make_snapshot(prompt_tokens=400, completion_tokens=100, tool_calls=1)
        second = catalog.make_snapshot(
            prompt_tokens=100,
            completion_tokens=50,
            tool_calls=2,
            model_name="custom-model",
        )
        merged = merge_usage([first, second])

        self.assertEqual(first.model, "demo-model")
        self.assertEqual(first.total_tokens, 500)
        self.assertEqual(first.estimated_cost_usd, 0.15)
        self.assertEqual(second.model, "custom-model")
        self.assertEqual(merged.model, "custom-model")
        self.assertEqual(merged.prompt_tokens, 500)
        self.assertEqual(merged.completion_tokens, 150)
        self.assertEqual(merged.total_tokens, 650)
        self.assertEqual(merged.estimated_cost_usd, 0.2)
        self.assertEqual(merged.tool_calls, 3)

    def test_merge_usage_returns_default_snapshot_for_empty_input(self) -> None:
        merged = merge_usage([])
        self.assertEqual(merged.model, "unconfigured")
        self.assertEqual(merged.total_tokens, 0)
        self.assertEqual(merged.tool_calls, 0)


class ToolRegistryTests(unittest.TestCase):
    def test_tool_registry_exposes_expected_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_dir = root / "agents"
            skill_dir = root / "skills"
            agent_dir.mkdir()
            skill_dir.mkdir()
            (agent_dir / "goal-runner.md").write_text(
                """---
name: goal-runner
description: Main agent
---
Prompt
""",
                encoding="utf-8",
            )
            (skill_dir / "inspect.md").write_text(
                """---
name: inspect
description: Inspect things
---
Instructions
""",
                encoding="utf-8",
            )
            config_path = root / "mcp_servers.yaml"
            config_path.write_text(
                """servers:
  - name: docs
    command: python
    args: [server.py]
""",
                encoding="utf-8",
            )

            agent_registry = AgentRegistry(agent_dir)
            agent_registry.refresh()
            skill_registry = SkillRegistry(skill_dir)
            skill_registry.refresh()
            mcp_bridge = MCPBridge(config_path)
            registry = ToolRegistry(
                ToolContext(
                    agent_registry=agent_registry,
                    skill_registry=skill_registry,
                    mcp_bridge=mcp_bridge,
                    memory_snapshot=lambda: "memory line",
                )
            )

            self.assertEqual(
                registry.names(),
                ["clock", "echo", "list_agents", "list_skills", "mcp_status", "memory_snapshot"],
            )
            self.assertEqual(registry.execute("echo").output, "(empty input)")
            self.assertIn("goal-runner", registry.execute("list_agents").output)
            self.assertIn("inspect", registry.execute("list_skills").output)
            self.assertEqual(registry.execute("memory_snapshot").output, "memory line")
            self.assertIn("docs", registry.execute("mcp_status").output)
            self.assertTrue(registry.execute("clock").output)
            unknown = registry.execute("missing")
            self.assertFalse(unknown.ok)
            self.assertEqual(unknown.output, "Unknown tool: missing")


class MemoryAndHookTests(unittest.TestCase):
    def test_memory_store_round_trips_events_memories_and_usage(self) -> None:
        with TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "runtime.db")
            run_id = store.start_run("inspect state")
            store.record_event(run_id, "system", "started")
            store.append_memory(run_id, MemoryKind.WORKING, "first note")
            store.append_memory(run_id, MemoryKind.SUMMARY, "second note")
            store.record_usage(
                run_id,
                UsageSnapshot(
                    model="heuristic-fallback",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    estimated_cost_usd=0.01,
                    tool_calls=1,
                ),
            )
            store.finish_run(run_id)

            memories = store.load_memories(run_id)
            events = store.load_events(run_id)
            usage = store.load_usage_events(run_id)
            self.assertEqual([item.content for item in memories], ["first note", "second note"])
            self.assertEqual(events[0].message, "started")
            self.assertEqual(usage[0].total_tokens, 15)
            self.assertEqual(
                store.recent_memory_text(run_id, limit=1),
                "[summary] second note",
            )
            store.connection.close()

    def test_hook_manager_records_after_events_into_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "runtime.db")
            run_id = store.start_run("run hooks")
            hooks = HookManager(store)
            agent = AgentCard(
                name="goal-runner",
                description="Test agent",
                hooks={
                    "before_plan": ["{agent} planning {details}"],
                    "after_tool": ["{agent} used tool {details}"],
                },
            )

            hooks.fire(run_id, agent, "before_plan", details="iteration 1")
            hooks.fire(run_id, agent, "after_tool", details="clock: 2026-05-09T00:00:00+00:00")

            events = store.load_events(run_id)
            memories = store.load_memories(run_id)
            self.assertEqual([event.role for event in events], ["hook:before_plan", "hook:after_tool"])
            self.assertEqual(len(memories), 1)
            self.assertEqual(
                memories[0].content,
                "goal-runner used tool clock: 2026-05-09T00:00:00+00:00",
            )
            store.connection.close()


if __name__ == "__main__":
    unittest.main()
