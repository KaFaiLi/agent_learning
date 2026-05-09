from pathlib import Path

from agent_learning.config import AzureOpenAISettings, PricingSettings, RuntimeSettings
from agent_learning.engine import build_engine
from agent_learning.llm import StubScript
from agent_learning.models import AcceptanceSpec, BudgetState
from agent_learning.ralph import run_until_goal


def _settings(workspace: Path, agent_dir: Path) -> RuntimeSettings:
    return RuntimeSettings(
        workspace=workspace,
        agent_dir=agent_dir,
        skill_dir=workspace / "skills",
        mcp_config_path=workspace / "missing.yaml",
        settings_path=workspace / "missing.json",
        runtime_dir=workspace / ".agent_runtime",
        azure=AzureOpenAISettings(),
        pricing=PricingSettings(),
        max_iterations=5,
    )


def _agent_md(agent_dir: Path, *, verify: str, rubric: str | None = None, max_outer: int = 3) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    rubric_block = f'  rubric: |\n    {rubric}\n' if rubric else ""
    (agent_dir / "goal-runner.md").write_text(
        f"""---
name: goal-runner
description: tester
tools: [bash]
max_iterations: 2
acceptance:
  verify_command: "{verify}"
{rubric_block}  max_outer_iterations: {max_outer}
---
Run things.
""",
        encoding="utf-8",
    )


def test_ralph_stops_when_verify_passes(tmp_path):
    workspace = tmp_path
    (workspace / ".agent_runtime").mkdir()
    agent_dir = workspace / "agents"
    _agent_md(agent_dir, verify="true")
    settings = _settings(workspace, agent_dir)

    script = StubScript(turns=[([], "Tried.")] * 6)
    engine = build_engine(settings, llm=script)
    report = run_until_goal(engine, goal="Anything", budget=BudgetState(run_budget_usd=10.0))
    assert report.verdict is not None
    assert report.verdict.met is True


def test_ralph_iterates_when_verify_fails(tmp_path):
    workspace = tmp_path
    (workspace / ".agent_runtime").mkdir()
    agent_dir = workspace / "agents"
    _agent_md(agent_dir, verify="false", max_outer=2)
    settings = _settings(workspace, agent_dir)

    script = StubScript(turns=[([], "Tried.")] * 6)
    engine = build_engine(settings, llm=script)
    report = run_until_goal(engine, goal="Anything", budget=BudgetState(run_budget_usd=10.0))
    assert report.verdict is not None
    assert report.verdict.met is False
