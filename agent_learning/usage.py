from __future__ import annotations

from agent_learning.config import PricingSettings
from agent_learning.models import UsageSnapshot


class PricingCatalog:
    def __init__(self, settings: PricingSettings) -> None:
        self.settings = settings

    def estimate(self, prompt_tokens: int, completion_tokens: int) -> float:
        s = self.settings
        return (
            (prompt_tokens / 1000.0) * s.input_price_per_1k
            + (completion_tokens / 1000.0) * s.output_price_per_1k
        )

    def snapshot(self, *, prompt_tokens: int, completion_tokens: int, model: str | None = None) -> UsageSnapshot:
        return UsageSnapshot(
            model=model or self.settings.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=self.estimate(prompt_tokens, completion_tokens),
        )


def merge_usage(snapshots: list[UsageSnapshot]) -> UsageSnapshot:
    if not snapshots:
        return UsageSnapshot()
    out = UsageSnapshot(model=snapshots[0].model)
    for snap in snapshots:
        out.prompt_tokens += snap.prompt_tokens
        out.completion_tokens += snap.completion_tokens
        out.estimated_cost_usd += snap.estimated_cost_usd
        out.tool_calls += snap.tool_calls
    return out
