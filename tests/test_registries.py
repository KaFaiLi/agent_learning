from agent_learning.agents import AgentRegistry
from agent_learning.skills import SkillRegistry


def _write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_agent_registry_loads_frontmatter(tmp_path):
    _write(
        tmp_path / "demo.md",
        """---
name: demo
description: A demo agent.
tools: [read_file, bash]
subagents: [reviewer]
skills: [workspace-etiquette]
hooks:
  PreToolUse:
    - "[hook] {tool}"
acceptance:
  verify_command: "pytest -q"
  max_outer_iterations: 3
---
You are demo.
""",
    )
    registry = AgentRegistry(tmp_path)
    card = registry.get("demo")
    assert card is not None
    assert card.tools == ["read_file", "bash"]
    assert card.subagents == ["reviewer"]
    assert "PreToolUse" in {ev.value for ev in card.hooks}
    assert card.acceptance is not None
    assert card.acceptance.verify_command == "pytest -q"
    assert card.system_prompt.startswith("You are demo.")


def test_skill_registry_triggers(tmp_path):
    _write(
        tmp_path / "py.md",
        """---
name: python-debug
description: debug python
triggers: [pytest, traceback]
---
Heuristics for fixing failing tests.
""",
    )
    _write(
        tmp_path / "etiquette.md",
        """---
name: etiquette
description: always-on
---
body
""",
    )
    registry = SkillRegistry(tmp_path)
    selected = registry.select(declared=["etiquette"], goal="Fix the failing pytest", recent_text="")
    names = {s.name for s in selected}
    assert "etiquette" in names
    assert "python-debug" in names

    rendered = registry.render(selected)
    assert "## Skills" in rendered
    assert "python-debug" in rendered
