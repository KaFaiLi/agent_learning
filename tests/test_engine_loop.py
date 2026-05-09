from pathlib import Path

from agent_learning.config import AzureOpenAISettings, PricingSettings, RuntimeSettings
from agent_learning.engine import build_engine
from agent_learning.llm import StubScript
from agent_learning.models import BudgetState, ToolCall


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


def _agent(agent_dir: Path) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "goal-runner.md").write_text(
        """---
name: goal-runner
description: tester
tools: [read_file, edit_file]
max_iterations: 4
---
You write files.
""",
        encoding="utf-8",
    )


def test_engine_runs_tool_calls_and_finishes(tmp_path):
    workspace = tmp_path
    workspace.joinpath(".agent_runtime").mkdir()
    workspace.joinpath("note.txt").write_text("hello", encoding="utf-8")
    agent_dir = workspace / "agents"
    _agent(agent_dir)
    settings = _settings(workspace, agent_dir)
    settings.runtime_dir.mkdir(exist_ok=True)

    script = StubScript(
        turns=[
            ([ToolCall(id="t1", name="read_file", arguments={"path": "note.txt"})], None),
            (
                [ToolCall(id="t2", name="edit_file", arguments={"path": "note.txt", "old_string": "hello", "new_string": "hello world"})],
                None,
            ),
            ([], "Done. Note now reads 'hello world'."),
        ]
    )
    engine = build_engine(settings, llm=script)
    captured: list[dict] = []
    engine.add_listener(captured.append)
    report = engine.run_goal("Update note.txt", agent_name="goal-runner", budget=BudgetState(run_budget_usd=10))

    assert "hello world" in workspace.joinpath("note.txt").read_text(encoding="utf-8")
    assert "Done" in report.final_message
    kinds = [e["kind"] for e in captured]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "agent_stop" in kinds
