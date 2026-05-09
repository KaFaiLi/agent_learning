from __future__ import annotations

import argparse
from importlib.metadata import version as _pkg_version
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text

from agent_learning.config import RuntimeSettings
from agent_learning.engine import RuntimeEngine
from agent_learning.models import RunReport


# ──────────────────────────────────────────────────────────────────────────────
# Helper: package version
# ──────────────────────────────────────────────────────────────────────────────

def _version() -> str:
    try:
        return _pkg_version("agent-learning")
    except Exception:
        return "0.1.0"


# ──────────────────────────────────────────────────────────────────────────────
# ConsoleCLI – Claude-Code-style interactive REPL
# ──────────────────────────────────────────────────────────────────────────────

_HELP_TEXT = """\
[bold]Available commands[/bold]

  [green]/help[/green]    Show this help message
  [green]/status[/green]  Show session statistics and configuration
  [green]/clear[/green]   Clear the terminal screen
  [green]/exit[/green]    End the session and quit  ([dim]Ctrl-D / Ctrl-C also work[/dim])

Any other input is sent as a message to the agent.
"""

_ROLE_STYLES: dict[str, str] = {
    "system": "dim",
    "user": "bold green",
    "assistant": "bold cyan",
    "tool": "yellow",
}


class ConsoleCLI:
    """Interactive multi-turn REPL with a Claude-Code-inspired terminal UI."""

    def __init__(self, engine: RuntimeEngine, initial_message: str | None = None) -> None:
        self.engine = engine
        self.initial_message = initial_message
        self.console = Console(highlight=False)
        self._session_id: str | None = None
        self._events_shown: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._turn: int = 0

    # ── public entry-point ────────────────────────────────────────────────────

    def run(self) -> None:
        self._print_banner()

        if self.initial_message:
            self._process_message(self.initial_message)

        while True:
            try:
                user_input = self._prompt()
            except (EOFError, KeyboardInterrupt):
                self._print_goodbye()
                break

            stripped = user_input.strip()
            if not stripped:
                continue

            if stripped.startswith("/"):
                if self._handle_command(stripped):
                    break
                continue

            self._process_message(stripped)

    # ── banner / farewell ─────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        ver = _version()
        body = Text.assemble(
            ("✻ agent-learning", "bold green"),
            (f"  v{ver}\n\n", "dim green"),
            ("  /help", "bold white"),
            (" for help  ·  ", "dim"),
            ("/exit", "bold white"),
            (" to quit", "dim"),
        )
        self.console.print(Panel(body, border_style="green", padding=(0, 1)))
        self.console.print()

    def _print_goodbye(self) -> None:
        self.console.print()
        if self._session_id:
            self.engine.end_session(self._session_id)
        self.console.print(
            Panel(
                Text.assemble(
                    ("Session ended. ", "dim"),
                    (f"Turns: {self._turn}  ", "dim"),
                    (f"Tokens: {self._total_tokens:,}  ", "dim"),
                    (f"Cost: ${self._total_cost:.4f}", "dim"),
                ),
                border_style="dim",
                padding=(0, 1),
            )
        )

    # ── prompt ────────────────────────────────────────────────────────────────

    def _prompt(self) -> str:
        self.console.print()
        self.console.print(Text("❯ ", style="bold green"), end="")
        return input()

    # ── message processing ────────────────────────────────────────────────────

    def _process_message(self, message: str) -> None:
        self._turn += 1

        # Start a new session on the very first message.
        if self._session_id is None:
            self._session_id = self.engine.begin_session(message)
            self._events_shown = 0

        # Show spinner while the agent is working.
        spinner = Spinner("dots", text=Text("  Thinking…", style="dim"))
        with Live(spinner, console=self.console, transient=True):
            report = self.engine.run_session_turn(self._session_id, message)

        self._render_new_events(report)
        self._render_stats(report)

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_new_events(self, report: RunReport) -> None:
        new_events = report.events[self._events_shown:]
        self._events_shown = len(report.events)

        for event in new_events:
            role = event.role.lower()
            if role == "user":
                # Already shown by the user's own typing – skip.
                continue

            if role == "system":
                self.console.print(Text(f"  {event.message}", style="dim"))
            elif role == "assistant":
                self.console.print()
                # Render as Markdown so the agent's output looks polished.
                md = Markdown(event.message, justify="left")
                self.console.print(md)
            else:
                # Tool call or named agent event.
                label = event.role
                style = _ROLE_STYLES.get(role, "yellow")
                self.console.print(
                    Text.assemble(
                        ("  ⊕ ", style),
                        (f"[{label}]  ", "bold " + style),
                        (event.message, "default"),
                    )
                )

    def _render_stats(self, report: RunReport) -> None:
        self._total_tokens += report.usage.total_tokens
        self._total_cost += report.usage.estimated_cost_usd
        self.console.print()
        self.console.print(
            Rule(
                Text.assemble(
                    (f" {report.usage.total_tokens:,} tokens", "dim"),
                    ("  ·  ", "dim"),
                    (f"${report.usage.estimated_cost_usd:.4f}", "dim"),
                    ("  ·  ", "dim"),
                    (f"run {report.run_id}", "dim"),
                    (" ", "dim"),
                ),
                style="dim",
            )
        )

    # ── slash commands ────────────────────────────────────────────────────────

    def _handle_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True when the session should end."""
        cmd_lower = cmd.lower()

        if cmd_lower in {"/exit", "/quit", "/q"}:
            self._print_goodbye()
            return True

        if cmd_lower == "/help":
            self.console.print(Panel(_HELP_TEXT, border_style="dim", padding=(0, 1)))
            return False

        if cmd_lower == "/clear":
            self.console.clear()
            self._print_banner()
            return False

        if cmd_lower == "/status":
            self._print_status()
            return False

        self.console.print(Text(f"  Unknown command: {cmd}  (try /help)", style="dim red"))
        return False

    def _print_status(self) -> None:
        snapshot = self.engine.snapshot()
        lines: list[str] = [
            f"[bold]Session[/bold]  {self._session_id or '(not started)'}",
            f"[bold]Turns[/bold]    {self._turn}",
            f"[bold]Tokens[/bold]   {self._total_tokens:,}",
            f"[bold]Cost[/bold]     ${self._total_cost:.4f}",
            "",
            f"[bold]Agents[/bold]   {len(snapshot['agents'])}",
            f"[bold]Skills[/bold]   {len(snapshot['skills'])}",
            f"[bold]MCP[/bold]      {len(snapshot['mcp'])} server(s)",
            f"[bold]Azure[/bold]    {'configured' if snapshot['azure_configured'] else 'not configured'}",
            f"[bold]Model[/bold]    {self.engine.settings.pricing.model_name}",
        ]
        self.console.print(
            Panel("\n".join(lines), title="Status", border_style="dim", padding=(0, 1))
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-learning",
        description="Start the agent-learning interactive REPL.",
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
        "--tui",
        action="store_true",
        help="Launch the Textual TUI instead of the interactive REPL.",
    )
    parser.add_argument(
        "--goal",
        help=(
            "Optional opening message sent as the first turn of the REPL, "
            "or as the initial goal value when --tui is used."
        ),
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = RuntimeSettings.from_env(args.root)
    settings.ensure_storage_dirs()
    engine = RuntimeEngine(settings)

    if args.print_config:
        print(f"agent_dir={settings.agent_dir}")
        print(f"mcp_config_path={settings.mcp_config_path}")
        print(f"memory_store_path={settings.memory_store_path}")
        print(f"azure_configured={settings.azure.is_configured}")
        print(f"pricing_model_name={settings.pricing.model_name}")
        print(f"max_iterations={settings.max_iterations}")
        print(f"step_budget_usd={settings.default_step_budget_usd:.2f}")
        print(f"run_budget_usd={settings.default_run_budget_usd:.2f}")
        return

    goal = args.goal or settings.default_goal

    if args.tui:
        from agent_learning.tui.app import AgentLearningApp

        app = AgentLearningApp(engine=engine, initial_goal=goal)
        app.run()
        return

    cli = ConsoleCLI(engine=engine, initial_message=args.goal)
    cli.run()
