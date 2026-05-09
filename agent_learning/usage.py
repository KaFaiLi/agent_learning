from __future__ import annotations

from dataclasses import dataclass

from agent_learning.config import PricingSettings
from agent_learning.models import UsageSnapshot


@dataclass(slots=True)
class PricingCatalog:
    model_name: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float

    @classmethod
    def from_settings(cls, settings: PricingSettings) -> "PricingCatalog":
        return cls(
            model_name=settings.model_name,
            input_cost_per_1k_tokens=settings.input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=settings.output_cost_per_1k_tokens,
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_cost = (prompt_tokens / 1000.0) * self.input_cost_per_1k_tokens
        completion_cost = (completion_tokens / 1000.0) * self.output_cost_per_1k_tokens
        return round(prompt_cost + completion_cost, 6)

    def make_snapshot(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        tool_calls: int = 0,
        model_name: str | None = None,
    ) -> UsageSnapshot:
        total_tokens = prompt_tokens + completion_tokens
        return UsageSnapshot(
            model=model_name or self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=self.estimate_cost(prompt_tokens, completion_tokens),
            tool_calls=tool_calls,
        )


def merge_usage(snapshots: list[UsageSnapshot]) -> UsageSnapshot:
    if not snapshots:
        return UsageSnapshot()

    return UsageSnapshot(
        model=snapshots[-1].model,
        prompt_tokens=sum(item.prompt_tokens for item in snapshots),
        completion_tokens=sum(item.completion_tokens for item in snapshots),
        total_tokens=sum(item.total_tokens for item in snapshots),
        estimated_cost_usd=round(sum(item.estimated_cost_usd for item in snapshots), 6),
        tool_calls=sum(item.tool_calls for item in snapshots),
    )
