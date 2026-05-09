from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_learning.agent_defs import AgentRegistry
from agent_learning.mcp import MCPBridge
from agent_learning.skills import SkillRegistry


class RegistryParsingTests(unittest.TestCase):
    def test_agent_registry_parses_frontmatter_and_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            agent_dir = Path(tmp)
            (agent_dir / "alpha.md").write_text(
                """---
name: alpha
description: Alpha agent
tools:
  - clock
subagents:
  - helper
skills:
  - inspect
hooks:
  after_finish:
    - "{agent}: {details}"
memory_policy: Keep concise notes.
---
Follow the system prompt.
""",
                encoding="utf-8",
            )
            (agent_dir / "beta.md").write_text("Fallback body.", encoding="utf-8")

            registry = AgentRegistry(agent_dir)
            registry.refresh()

            self.assertEqual(registry.names(), ["alpha", "beta"])
            alpha = registry.get("alpha")
            beta = registry.get("beta")
            self.assertIsNotNone(alpha)
            self.assertIsNotNone(beta)
            assert alpha is not None
            assert beta is not None
            self.assertEqual(alpha.description, "Alpha agent")
            self.assertEqual(alpha.tools, ["clock"])
            self.assertEqual(alpha.subagents, ["helper"])
            self.assertEqual(alpha.skills, ["inspect"])
            self.assertEqual(alpha.hooks["after_finish"], ["{agent}: {details}"])
            self.assertEqual(alpha.memory_policy, "Keep concise notes.")
            self.assertEqual(alpha.system_prompt, "Follow the system prompt.")
            self.assertEqual(beta.description, "No description provided.")
            self.assertEqual(beta.system_prompt, "Fallback body.")

    def test_skill_registry_parses_frontmatter_and_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            (skill_dir / "inspect.md").write_text(
                """---
name: inspect
description: Inspect runtime state
---
Use read-only investigation.
""",
                encoding="utf-8",
            )
            (skill_dir / "fallback.md").write_text("Plain instructions.", encoding="utf-8")

            registry = SkillRegistry(skill_dir)
            registry.refresh()

            inspect = registry.get("inspect")
            fallback = registry.get("fallback")
            self.assertIsNotNone(inspect)
            self.assertIsNotNone(fallback)
            assert inspect is not None
            assert fallback is not None
            self.assertEqual(inspect.description, "Inspect runtime state")
            self.assertEqual(inspect.instructions, "Use read-only investigation.")
            self.assertEqual(fallback.name, "fallback")
            self.assertEqual(fallback.description, "No description provided.")
            self.assertEqual(fallback.instructions, "Plain instructions.")


class MCPBridgeTests(unittest.TestCase):
    def test_mcp_bridge_reports_missing_config(self) -> None:
        with TemporaryDirectory() as tmp:
            bridge = MCPBridge(Path(tmp) / "mcp_servers.yaml")
            self.assertEqual(
                bridge.describe(),
                "No MCP servers configured. Add mcp_servers.yaml to register stdio servers.",
            )
            self.assertEqual(bridge.list_servers(), [])

    def test_mcp_bridge_loads_servers_and_defaults_transport(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "mcp_servers.yaml"
            config_path.write_text(
                """servers:
  - name: docs
    command: python
    args:
      - serve.py
  - name: search
    command: node
    transport: websocket
""",
                encoding="utf-8",
            )

            bridge = MCPBridge(config_path)

            servers = bridge.list_servers()
            self.assertEqual(len(servers), 2)
            self.assertEqual(servers[0].name, "docs")
            self.assertEqual(servers[0].args, ["serve.py"])
            self.assertEqual(servers[0].transport, "stdio")
            self.assertEqual(servers[1].transport, "websocket")
            self.assertIn("- docs: python serve.py (stdio)", bridge.describe())
            self.assertIn("- search: node  (websocket)", bridge.describe())

    def test_mcp_bridge_reports_yaml_errors(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "mcp_servers.yaml"
            config_path.write_text("servers: [", encoding="utf-8")

            bridge = MCPBridge(config_path)

            self.assertIn("Invalid MCP config in mcp_servers.yaml:", bridge.describe())
            self.assertEqual(bridge.list_servers(), [])


if __name__ == "__main__":
    unittest.main()
