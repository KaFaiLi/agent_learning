from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(slots=True)
class AzureOpenAISettings:
    endpoint: str | None = None
    api_key: str | None = None
    deployment: str | None = None
    api_version: str = "2024-10-21"

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment)

    @classmethod
    def from_env(cls) -> "AzureOpenAISettings":
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or None,
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or None,
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT") or None,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )


@dataclass(slots=True)
class PricingSettings:
    model_name: str = "azure-deployment"
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0

    @classmethod
    def from_env(cls) -> "PricingSettings":
        return cls(
            model_name=os.getenv("AGENT_MODEL_NAME", "azure-deployment"),
            input_price_per_1k=float(os.getenv("AGENT_INPUT_PRICE_PER_1K", "0") or 0),
            output_price_per_1k=float(os.getenv("AGENT_OUTPUT_PRICE_PER_1K", "0") or 0),
        )


@dataclass(slots=True)
class RuntimeSettings:
    workspace: Path
    agent_dir: Path
    skill_dir: Path
    mcp_config_path: Path
    settings_path: Path
    runtime_dir: Path
    azure: AzureOpenAISettings = field(default_factory=AzureOpenAISettings)
    pricing: PricingSettings = field(default_factory=PricingSettings)
    default_agent: str = "goal-runner"
    default_goal: str = "Inspect the workspace and propose a useful next step."
    max_iterations: int = 12
    step_budget_usd: float = 0.5
    run_budget_usd: float = 5.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, *, dotenv: Path | None = None) -> "RuntimeSettings":
        if dotenv is None:
            dotenv = Path.cwd() / ".env"
        load_dotenv(dotenv)

        workspace = Path(os.getenv("AGENT_WORKSPACE", ".")).resolve()
        runtime_dir = workspace / ".agent_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        settings = cls(
            workspace=workspace,
            agent_dir=Path(os.getenv("AGENT_AGENT_DIR", str(workspace / "demo_agents"))).resolve(),
            skill_dir=Path(os.getenv("AGENT_SKILL_DIR", str(workspace / "demo_skills"))).resolve(),
            mcp_config_path=Path(os.getenv("AGENT_MCP_CONFIG", str(workspace / "mcp_servers.yaml"))).resolve(),
            settings_path=Path(os.getenv("AGENT_SETTINGS", str(workspace / "settings.json"))).resolve(),
            runtime_dir=runtime_dir,
            azure=AzureOpenAISettings.from_env(),
            pricing=PricingSettings.from_env(),
            default_agent=os.getenv("AGENT_DEFAULT_AGENT", "goal-runner"),
            default_goal=os.getenv("AGENT_DEFAULT_GOAL", "Inspect the workspace and propose a useful next step."),
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "12")),
            step_budget_usd=float(os.getenv("AGENT_STEP_BUDGET_USD", "0.5")),
            run_budget_usd=float(os.getenv("AGENT_RUN_BUDGET_USD", "5.0")),
        )
        if settings.settings_path.exists():
            try:
                settings.extra = json.loads(settings.settings_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                settings.extra = {}
        return settings

    def summary(self) -> str:
        lines = [
            f"workspace      : {self.workspace}",
            f"agent_dir      : {self.agent_dir}",
            f"skill_dir      : {self.skill_dir}",
            f"mcp_config     : {self.mcp_config_path}",
            f"settings.json  : {self.settings_path}",
            f"runtime_dir    : {self.runtime_dir}",
            f"azure_configured: {self.azure.is_configured}",
            f"deployment     : {self.azure.deployment or '-'}",
            f"default_agent  : {self.default_agent}",
            f"max_iterations : {self.max_iterations}",
            f"step_budget    : ${self.step_budget_usd}",
            f"run_budget     : ${self.run_budget_usd}",
        ]
        return "\n".join(lines)
