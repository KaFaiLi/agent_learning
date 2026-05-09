from __future__ import annotations

import json
import subprocess
from typing import Any

from agent_learning.engine import Engine
from agent_learning.models import (
    AcceptanceSpec,
    AgentCard,
    BudgetState,
    EvalVerdict,
    Message,
    RunReport,
)
from agent_learning.tools._sandbox import SandboxError, screen_command


def run_until_goal(
    engine: Engine,
    *,
    goal: str,
    agent_name: str | None = None,
    budget: BudgetState | None = None,
    acceptance_override: AcceptanceSpec | None = None,
) -> RunReport:
    settings = engine.deps.settings
    budget = budget or BudgetState(
        step_budget_usd=settings.step_budget_usd,
        run_budget_usd=settings.run_budget_usd,
    )

    agent = engine.deps.agents.get(agent_name or settings.default_agent)
    acceptance = acceptance_override or (agent.acceptance if agent else None)
    if acceptance is None:
        return engine.run_goal(goal, agent_name=agent_name, budget=budget)

    current_goal = goal
    last_report: RunReport | None = None
    final_verdict: EvalVerdict | None = None

    for outer in range(1, max(1, acceptance.max_outer_iterations) + 1):
        engine._emit({"kind": "ralph_iteration", "i": outer, "goal": current_goal})  # type: ignore[attr-defined]
        report = engine.run_goal(current_goal, agent_name=agent_name, budget=budget)
        last_report = report
        verdict = _evaluate(engine, acceptance, report)
        engine.deps.store.record(report.run_id, kind="verdict", role="ralph", payload=verdict.model_dump())
        engine._emit({"kind": "ralph_verdict", "i": outer, "verdict": verdict.model_dump()})  # type: ignore[attr-defined]
        final_verdict = verdict
        if verdict.met:
            break
        if budget.remaining_usd <= 0:
            break
        current_goal = (
            f"{goal}\n\nPrevious attempt finished with: {report.final_message}\n\n"
            f"Unresolved gaps:\n{verdict.gaps}\n\nAddress these and finish the goal."
        )

    assert last_report is not None
    return last_report.model_copy(update={"verdict": final_verdict})


def _evaluate(engine: Engine, acceptance: AcceptanceSpec, report: RunReport) -> EvalVerdict:
    settings = engine.deps.settings
    if acceptance.verify_command:
        try:
            screen_command(acceptance.verify_command)
        except SandboxError as exc:
            return EvalVerdict(met=False, gaps=str(exc), rationale="verify_command rejected by sandbox")
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", acceptance.verify_command],
                cwd=settings.workspace,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return EvalVerdict(met=False, gaps="verify_command timed out", rationale="timeout")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            return EvalVerdict(met=False, gaps=tail, rationale=f"verify_command exit {proc.returncode}")
        if not acceptance.rubric:
            return EvalVerdict(met=True, gaps="", rationale=f"verify_command exit 0")
    if acceptance.rubric:
        return _llm_rubric(engine, acceptance.rubric, report)
    return EvalVerdict(met=True, gaps="", rationale="no acceptance criteria")


def _llm_rubric(engine: Engine, rubric: str, report: RunReport) -> EvalVerdict:
    events = engine.deps.store.load(report.run_id)
    tail = events[-30:]
    transcript = "\n".join(
        f"[{e.role}] {e.kind}: {json.dumps(e.payload, ensure_ascii=False)[:300]}" for e in tail
    )
    system = (
        "You are an evaluator. Given a goal, an attempt's transcript, and acceptance criteria, "
        "decide if the goal is met. Reply with strict JSON: {\"met\": bool, \"gaps\": str, \"rationale\": str}."
    )
    user = (
        f"Goal: {report.goal}\n\n"
        f"Acceptance criteria (rubric):\n{rubric}\n\n"
        f"Final attempt message:\n{report.final_message}\n\n"
        f"Recent transcript:\n{transcript}"
    )
    try:
        reply = engine.deps.llm.chat(
            messages=[Message(role="system", content=system), Message(role="user", content=user)],
            tools=None,
        )
    except Exception as exc:
        return EvalVerdict(met=False, gaps=f"evaluator error: {exc}", rationale="llm failure")
    raw = reply.message.content or "{}"
    try:
        data = _extract_json(raw)
        return EvalVerdict.model_validate(data)
    except Exception:
        return EvalVerdict(met=False, gaps=raw[:500], rationale="non-json evaluator reply")


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        # strip a fenced block
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text[5:]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)
