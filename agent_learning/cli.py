from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_learning.config import RuntimeSettings
from agent_learning.engine import build_engine
from agent_learning.llm import AzureClient
from agent_learning.mcp import MCPBridge
from agent_learning.ralph import run_until_goal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-learning", description="Claude-Code-style TUI agent demo.")
    parser.add_argument("--goal", help="Goal text to run (with --no-tui or --once).")
    parser.add_argument("--agent", help="Agent name to use (defaults to settings).")
    parser.add_argument("--workspace", help="Workspace directory (default: cwd).")
    parser.add_argument("--no-tui", action="store_true", help="Run headless and stream events to stdout.")
    parser.add_argument("--once", action="store_true", help="Run a single goal then exit.")
    parser.add_argument("--ralph", action="store_true", help="Run with the until-goal-met outer loop.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved settings and exit.")
    return parser


def _apply_workspace_override(args: argparse.Namespace) -> None:
    if args.workspace:
        import os

        os.environ["AGENT_WORKSPACE"] = str(Path(args.workspace).resolve())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_workspace_override(args)
    settings = RuntimeSettings.from_env()

    if args.print_config:
        print(settings.summary())
        return 0

    if args.no_tui or args.once:
        return _run_headless(settings, args)
    return _run_tui(settings, args)


def _run_headless(settings: RuntimeSettings, args: argparse.Namespace) -> int:
    if not settings.azure.is_configured:
        print("Azure OpenAI is not configured; set AZURE_OPENAI_ENDPOINT/_API_KEY/_DEPLOYMENT.", file=sys.stderr)
        return 2
    goal = args.goal or settings.default_goal
    llm = AzureClient(settings.azure)
    mcp = MCPBridge(settings.mcp_config_path)
    engine = build_engine(settings, llm=llm, mcp_bridge=mcp)
    engine.add_listener(_print_event)
    try:
        if args.ralph:
            report = run_until_goal(engine, goal=goal, agent_name=args.agent)
        else:
            report = engine.run_goal(goal, agent_name=args.agent)
    finally:
        mcp.shutdown()
    print()
    print("=" * 60)
    print(f"Run {report.run_id} finished. iterations={report.iterations} cost=${report.usage.estimated_cost_usd:.4f} tokens={report.usage.total_tokens}")
    if report.verdict is not None:
        print(f"Verdict: met={report.verdict.met} gaps={report.verdict.gaps[:200]}")
    print(report.final_message)
    return 0


def _print_event(event: dict) -> None:
    kind = event.get("kind", "?")
    if kind == "assistant":
        print(f"\n[{event['agent']}] {event['content']}")
    elif kind == "tool_call":
        print(f"  -> {event['tool']}({_compact(event['arguments'])})")
    elif kind == "tool_result":
        marker = "ok" if event.get("ok") else "ERR"
        print(f"  <- [{marker}] {event['tool']}: {_compact(event['content'])}")
    elif kind == "iteration":
        print(f"\n--- iteration {event['i']} ({event['agent']}, depth={event.get('depth', 0)}) ---")
    elif kind == "ralph_verdict":
        v = event["verdict"]
        print(f"\n[ralph] iter={event['i']} met={v['met']} gaps={v.get('gaps', '')[:200]}")
    elif kind == "agent_start":
        print(f"\n>>> agent_start {event['agent']} (depth={event['depth']})")
    elif kind == "agent_stop":
        print(f"<<< agent_stop {event['agent']} (depth={event['depth']})")


def _compact(value, *, limit: int = 240) -> str:
    s = value if isinstance(value, str) else repr(value)
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _run_tui(settings: RuntimeSettings, args: argparse.Namespace) -> int:
    from agent_learning.tui.app import AgentLearningApp

    app = AgentLearningApp(settings=settings, initial_goal=args.goal or settings.default_goal, ralph=args.ralph, agent_name=args.agent)
    app.run()
    return 0
