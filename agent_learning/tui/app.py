from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, ProgressBar, RichLog, Static

from agent_learning.config import RuntimeSettings
from agent_learning.engine import Engine, build_engine
from agent_learning.llm import AzureClient
from agent_learning.mcp import MCPBridge
from agent_learning.models import BudgetState
from agent_learning.ralph import run_until_goal


class AgentLearningApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; padding: 0 1; }
    #goal { width: 1fr; }
    #content { height: 1fr; }
    #log { width: 3fr; border: solid $primary; margin-right: 1; }
    #sidebar { width: 2fr; }
    .panel { border: solid $secondary; padding: 1; margin-bottom: 1; height: auto; }
    #budget-bar { width: 1fr; }
    """

    BINDINGS = [
        ("ctrl+r", "run_goal", "Run"),
        ("ctrl+l", "refresh", "Refresh"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, *, settings: RuntimeSettings, initial_goal: str, ralph: bool = False, agent_name: str | None = None) -> None:
        super().__init__()
        self.settings = settings
        self.initial_goal = initial_goal
        self.ralph = ralph
        self.agent_name = agent_name
        self.engine: Engine | None = None
        self.mcp: MCPBridge | None = None
        self._event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._running = False

    # ---- composition -----------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            yield Input(value=self.initial_goal, placeholder="Goal or /command (try /help)", id="goal")
            with Horizontal(id="content"):
                yield RichLog(id="log", wrap=True, markup=True, highlight=True)
                with Vertical(id="sidebar"):
                    yield Static(id="runtime", classes="panel")
                    yield Static(id="agents", classes="panel")
                    yield Static(id="skills", classes="panel")
                    yield Static(id="mcp", classes="panel")
                    yield Static("Budget", classes="panel", id="budget-label")
                    yield ProgressBar(total=100, id="budget-bar")
        yield Footer()

    # ---- lifecycle -------------------------------------------------

    async def on_mount(self) -> None:
        self._init_engine()
        self._refresh_panels()
        self._log("[b]agent-learning[/b] ready. Type a goal or /help.")
        self.set_interval(0.1, self._drain_events)

    def _init_engine(self) -> None:
        if not self.settings.azure.is_configured:
            self._log("[red]Azure OpenAI not configured.[/red] Set AZURE_OPENAI_ENDPOINT/_API_KEY/_DEPLOYMENT to run.")
            return
        try:
            llm = AzureClient(self.settings.azure)
        except Exception as exc:
            self._log(f"[red]Failed to init Azure client: {exc}[/red]")
            return
        self.mcp = MCPBridge(self.settings.mcp_config_path)
        self.engine = build_engine(self.settings, llm=llm, mcp_bridge=self.mcp)
        self.engine.add_listener(lambda e: self._event_queue.put(e))

    def on_unmount(self) -> None:
        if self.mcp is not None:
            self.mcp.shutdown()

    # ---- input handling --------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(text)
            return
        await self._kick_run(text)

    async def action_run_goal(self) -> None:
        widget = self.query_one("#goal", Input)
        text = widget.value.strip() or self.initial_goal
        widget.value = ""
        await self._kick_run(text)

    async def action_refresh(self) -> None:
        if self.engine is not None:
            self.engine.refresh()
        self._refresh_panels()
        self._log("Refreshed agents/skills/MCP.")

    async def _kick_run(self, goal: str) -> None:
        if self.engine is None:
            self._log("[red]Engine not ready.[/red]")
            return
        if self._running:
            self._log("[yellow]A run is already in progress.[/yellow]")
            return
        self._running = True
        self._log(f"[b]>> goal[/b]: {goal}{' (ralph)' if self.ralph else ''}")
        budget = BudgetState(
            step_budget_usd=self.settings.step_budget_usd,
            run_budget_usd=self.settings.run_budget_usd,
        )
        self._update_budget(budget)

        def worker() -> None:
            try:
                if self.ralph:
                    report = run_until_goal(self.engine, goal=goal, agent_name=self.agent_name, budget=budget)
                else:
                    report = self.engine.run_goal(goal, agent_name=self.agent_name, budget=budget)
                self._event_queue.put({"kind": "_report", "report": report.model_dump(), "budget": budget.model_dump()})
            except Exception as exc:
                self._event_queue.put({"kind": "_error", "error": f"{type(exc).__name__}: {exc}"})

        threading.Thread(target=worker, daemon=True).start()

    # ---- commands --------------------------------------------------

    def _handle_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        cmd = parts[0][1:].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd in ("help", "?"):
            self._log(
                "Commands: /goal <text>, /agent <name>, /ralph on|off, /agents, /skills, /mcp, /budget, /clear, /quit"
            )
        elif cmd == "goal":
            self.query_one("#goal", Input).value = rest
        elif cmd == "agent":
            self.agent_name = rest or None
            self._log(f"Active agent: {self.agent_name or '(default)'}")
        elif cmd == "ralph":
            self.ralph = rest.strip().lower() not in ("off", "false", "0", "no")
            self._log(f"Ralph mode: {'on' if self.ralph else 'off'}")
        elif cmd == "agents":
            self._refresh_panels()
            self._log(self._panel_text("agents") or "(no agents)")
        elif cmd == "skills":
            self._refresh_panels()
            self._log(self._panel_text("skills") or "(no skills)")
        elif cmd == "mcp":
            self._refresh_panels()
            self._log(self._panel_text("mcp") or "(no MCP)")
        elif cmd == "budget":
            self._log(f"Budget: step=${self.settings.step_budget_usd} run=${self.settings.run_budget_usd}")
        elif cmd == "clear":
            self.query_one("#log", RichLog).clear()
        elif cmd == "quit":
            self.exit()
        else:
            self._log(f"Unknown command: /{cmd}")

    # ---- event drain -----------------------------------------------

    def _drain_events(self) -> None:
        drained = 0
        while drained < 50:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                return
            drained += 1
            self._render_event(event)

    def _render_event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind == "_report":
            report = event["report"]
            budget = event["budget"]
            self._update_budget(BudgetState(**budget))
            verdict = report.get("verdict")
            verdict_text = ""
            if verdict is not None:
                verdict_text = f" verdict.met={verdict['met']}"
            self._log(
                f"[green]done[/green] iters={report['iterations']} cost=${report['usage']['estimated_cost_usd']:.4f}{verdict_text}"
            )
            self._log(f"[b]final[/b]: {report['final_message']}")
            self._running = False
        elif kind == "_error":
            self._log(f"[red]error[/red]: {event['error']}")
            self._running = False
        elif kind == "iteration":
            self._log(f"--- iter {event['i']} ({event['agent']}, depth={event.get('depth', 0)})")
        elif kind == "assistant":
            self._log(f"[cyan][{event['agent']}][/cyan] {event['content']}")
        elif kind == "tool_call":
            args = _short(event.get("arguments"))
            self._log(f"  [magenta]→[/magenta] {event['tool']}({args})")
        elif kind == "tool_result":
            marker = "[green]ok[/green]" if event.get("ok") else "[red]err[/red]"
            self._log(f"  [magenta]←[/magenta] {marker} {event['tool']}: {_short(event['content'])}")
        elif kind == "agent_start":
            self._log(f">>> [b]{event['agent']}[/b] started (depth={event['depth']})")
        elif kind == "agent_stop":
            self._log(f"<<< [b]{event['agent']}[/b] stopped (depth={event['depth']})")
        elif kind == "ralph_iteration":
            self._log(f"[yellow][ralph][/yellow] outer iteration {event['i']}")
        elif kind == "ralph_verdict":
            v = event["verdict"]
            colour = "green" if v["met"] else "yellow"
            self._log(f"[{colour}][ralph verdict][/{colour}] met={v['met']} gaps={_short(v.get('gaps'))}")

    # ---- panels & utility ------------------------------------------

    def _refresh_panels(self) -> None:
        if self.engine is None:
            return
        snap = self.engine.snapshot()
        self.query_one("#runtime", Static).update(
            "\n".join(
                [
                    "Runtime",
                    f"workspace: {snap['workspace']}",
                    f"azure: {'on' if snap['azure_configured'] else 'off'}",
                    f"tools: {len(snap['tools'])}",
                ]
            )
        )
        agents_text = "\n".join(["Agents"] + [f"- {a['name']}: {a['description']}" for a in snap["agents"]] or ["Agents", "(none)"])
        self.query_one("#agents", Static).update(agents_text)
        skills_text = "\n".join(["Skills"] + [f"- {s['name']}" for s in snap["skills"]] or ["Skills", "(none)"])
        self.query_one("#skills", Static).update(skills_text)
        self.query_one("#mcp", Static).update("MCP\n" + (snap["mcp"] or "(none)"))

    def _panel_text(self, panel_id: str) -> str:
        widget = self.query_one(f"#{panel_id}", Static)
        return str(widget.renderable)

    def _update_budget(self, budget: BudgetState) -> None:
        bar = self.query_one("#budget-bar", ProgressBar)
        if budget.run_budget_usd > 0:
            pct = min(100.0, 100.0 * budget.spent_usd / budget.run_budget_usd)
        else:
            pct = 0.0
        bar.update(progress=pct)
        self.query_one("#budget-label", Static).update(
            f"Budget: ${budget.spent_usd:.4f}/${budget.run_budget_usd:.2f}"
        )

    def _log(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)


def _short(value: Any, *, limit: int = 220) -> str:
    s = value if isinstance(value, str) else repr(value)
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else s[: limit - 3] + "..."
