from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

from agent_learning.engine import RuntimeEngine
from agent_learning.models import RunReport


class AgentLearningApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
        padding: 1 2;
    }

    #controls {
        height: auto;
        padding-bottom: 1;
    }

    #goal-input {
        width: 1fr;
        margin-bottom: 1;
    }

    #content {
        height: 1fr;
    }

    #event-log {
        width: 3fr;
        border: solid $primary;
        margin-right: 1;
    }

    #sidebar {
        width: 2fr;
    }

    .panel {
        border: solid $secondary;
        padding: 1;
        margin-bottom: 1;
        height: auto;
    }
    """

    BINDINGS = [
        ("ctrl+r", "run_goal", "Run Goal"),
        ("ctrl+l", "refresh_state", "Refresh"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, *, engine: RuntimeEngine, initial_goal: str) -> None:
        super().__init__()
        self.engine = engine
        self.initial_goal = initial_goal
        self.last_report: RunReport | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            yield Input(value=self.initial_goal, placeholder="Enter a goal", id="goal-input")
            with Horizontal(id="controls"):
                yield Button("Run Goal", id="run-goal", variant="primary")
                yield Button("Refresh", id="refresh")
            with Horizontal(id="content"):
                yield RichLog(id="event-log", wrap=True, highlight=True, markup=True)
                with Vertical(id="sidebar"):
                    yield Static(id="runtime-panel", classes="panel")
                    yield Static(id="agent-panel", classes="panel")
                    yield Static(id="skill-panel", classes="panel")
                    yield Static(id="memory-panel", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_state()
        self._log("Runtime ready. Enter a goal and run the loop.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-goal":
            self.action_run_goal()
        elif event.button.id == "refresh":
            self.action_refresh_state()

    def action_refresh_state(self) -> None:
        self._refresh_state()
        self._log("Refreshed agents, skills, and MCP configuration.")

    def action_run_goal(self) -> None:
        self.run_worker(self._run_goal(), exclusive=True)

    async def _run_goal(self) -> None:
        goal_widget = self.query_one("#goal-input", Input)
        goal = goal_widget.value.strip() or self.initial_goal
        self._log(f"[b]Goal[/b]: {goal}")
        report = await asyncio.to_thread(self.engine.run_goal, goal)
        self.last_report = report
        self._render_report(report)
        self._log(f"[green]Completed[/green]: {report.final_message}")

    def _refresh_state(self) -> None:
        snapshot = self.engine.snapshot()
        agent_panel = self.query_one("#agent-panel", Static)
        skill_panel = self.query_one("#skill-panel", Static)
        runtime_panel = self.query_one("#runtime-panel", Static)
        memory_panel = self.query_one("#memory-panel", Static)

        agent_lines = ["Agents"]
        agent_lines.extend(
            f"- {agent.name}: tools={len(agent.tools)} subagents={len(agent.subagents)} hooks={len(agent.hooks)}"
            for agent in snapshot["agents"]
        )
        if len(agent_lines) == 1:
            agent_lines.append("- No markdown agents loaded")

        skill_lines = ["Skills"]
        skill_lines.extend(f"- {skill.name}: {skill.description}" for skill in snapshot["skills"])
        if len(skill_lines) == 1:
            skill_lines.append("- No skills loaded")

        runtime_lines = ["Runtime"]
        runtime_lines.append(f"- Azure configured: {snapshot['azure_configured']}")
        runtime_lines.append(f"- MCP servers: {len(snapshot['mcp'])}")
        runtime_lines.append(f"- Memory Store: {self.engine.settings.memory_store_path}")

        memory_lines = ["Latest Run"]
        if self.last_report is None:
            memory_lines.append("- No run yet")
        else:
            memory_lines.append(f"- Run ID: {self.last_report.run_id}")
            memory_lines.append(f"- Memories: {len(self.last_report.memories)}")
            memory_lines.append(f"- Tokens: {self.last_report.usage.total_tokens}")
            memory_lines.append(f"- Cost USD: {self.last_report.usage.estimated_cost_usd:.4f}")

        runtime_panel.update("\n".join(runtime_lines))
        agent_panel.update("\n".join(agent_lines))
        skill_panel.update("\n".join(skill_lines))
        memory_panel.update("\n".join(memory_lines))

    def _render_report(self, report: RunReport) -> None:
        event_log = self.query_one("#event-log", RichLog)
        event_log.clear()
        for event in report.events:
            self._log(f"[{event.role}] {event.message}")
        self._refresh_state()

    def _log(self, message: str) -> None:
        event_log = self.query_one("#event-log", RichLog)
        event_log.write(message)
