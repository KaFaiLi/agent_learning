from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(slots=True)
class AzureOpenAISettings:
    endpoint: str | None
    api_key: str | None
    deployment: str | None
    api_version: str = "2024-10-21"

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment)

    @classmethod
    def from_env(cls) -> "AzureOpenAISettings":
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )


@dataclass(slots=True)
class PricingSettings:
    model_name: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float

    @classmethod
    def from_env(cls) -> "PricingSettings":
        return cls(
            model_name=os.getenv("AGENT_LEARNING_MODEL_NAME", "azure-deployment"),
            input_cost_per_1k_tokens=float(
                os.getenv("AGENT_LEARNING_INPUT_PRICE_PER_1K_TOKENS", "0.0")
            ),
            output_cost_per_1k_tokens=float(
                os.getenv("AGENT_LEARNING_OUTPUT_PRICE_PER_1K_TOKENS", "0.0")
            ),
        )


@dataclass(slots=True)
class RuntimeSettings:
    root_dir: Path
    azure: AzureOpenAISettings
    pricing: PricingSettings
    agent_dir: Path
    skill_dir: Path
    mcp_config_path: Path
    memory_store_path: Path
    default_goal: str
    max_iterations: int
    default_step_budget_usd: float
    default_run_budget_usd: float

    @classmethod
    def from_env(cls, root: Path | None = None) -> "RuntimeSettings":
        base_dir = root or Path.cwd()
        load_dotenv(base_dir / ".env", override=False)
        storage_root = base_dir / ".agent_learning"
        configured_memory_path = os.getenv("AGENT_LEARNING_MEMORY_PATH") or os.getenv(
            "AGENT_LEARNING_MEMORY_DB"
        )
        return cls(
            root_dir=base_dir,
            azure=AzureOpenAISettings.from_env(),
            pricing=PricingSettings.from_env(),
            agent_dir=Path(os.getenv("AGENT_LEARNING_AGENT_DIR", str(base_dir / "demo_agents"))),
            skill_dir=Path(os.getenv("AGENT_LEARNING_SKILL_DIR", str(base_dir / "demo_skills"))),
            mcp_config_path=Path(
                os.getenv("AGENT_LEARNING_MCP_CONFIG", str(base_dir / "mcp_servers.yaml"))
            ),
            memory_store_path=cls._normalize_memory_store_path(
                Path(configured_memory_path) if configured_memory_path else storage_root / "runtime.parquet"
            ),
            default_goal=os.getenv(
                "AGENT_LEARNING_DEFAULT_GOAL",
                "Inspect the repository, explain the current state, and propose the next useful implementation step.",
            ),
            max_iterations=int(os.getenv("AGENT_LEARNING_MAX_ITERATIONS", "6")),
            default_step_budget_usd=float(os.getenv("AGENT_LEARNING_STEP_BUDGET_USD", "0.50")),
            default_run_budget_usd=float(os.getenv("AGENT_LEARNING_RUN_BUDGET_USD", "5.00")),
        )

    def ensure_storage_dirs(self) -> None:
        self.memory_store_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_memory_store_path(path: Path) -> Path:
        if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            return path.with_suffix(".parquet")
        return path
