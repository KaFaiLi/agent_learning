from __future__ import annotations

import argparse
from pathlib import Path

from agent_learning.config import RuntimeSettings
from agent_learning.engine import RuntimeEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-learning",
        description="Start the agent-learning demo application.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved runtime configuration and exit.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve agent and storage paths.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single console session instead of launching the TUI.",
    )
    parser.add_argument(
        "--goal",
        help="Goal used for --once or as the initial TUI value.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = RuntimeSettings.from_env(args.root)
    settings.ensure_storage_dirs()
    engine = RuntimeEngine(settings)

    if args.print_config:
        print(f"agent_dir={settings.agent_dir}")
        print(f"mcp_config_path={settings.mcp_config_path}")
        print(f"memory_db_path={settings.memory_db_path}")
        print(f"azure_configured={settings.azure.is_configured}")
        print(f"pricing_model_name={settings.pricing.model_name}")
        print(f"max_iterations={settings.max_iterations}")
        print(f"step_budget_usd={settings.default_step_budget_usd:.2f}")
        print(f"run_budget_usd={settings.default_run_budget_usd:.2f}")
        return

    goal = args.goal or settings.default_goal

    if args.once:
        report = engine.run_goal(goal)
        print(f"Run {report.run_id}: {report.final_message}")
        print(f"Agents: {', '.join(agent.name for agent in report.agents) or 'none'}")
        print(f"Memories: {len(report.memories)}")
        print(
            "Usage: "
            f"model={report.usage.model} total_tokens={report.usage.total_tokens} "
            f"estimated_cost_usd={report.usage.estimated_cost_usd:.4f} "
            f"tool_calls={report.usage.tool_calls}"
        )
        return

    from agent_learning.tui.app import AgentLearningApp

    app = AgentLearningApp(engine=engine, initial_goal=goal)
    app.run()
