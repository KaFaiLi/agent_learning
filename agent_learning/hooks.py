from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from agent_learning.models import HookEvent, HookResponse, HookSpec


@dataclass
class HookBundle:
    """Resolved set of hooks merged from settings.json and an agent card."""

    by_event: dict[HookEvent, list[HookSpec]] = field(default_factory=dict)

    @classmethod
    def from_sources(
        cls,
        *,
        settings_hooks: dict[str, list[dict]] | None,
        agent_hooks: dict[HookEvent, list[HookSpec]] | None,
    ) -> "HookBundle":
        bundle = cls()
        if settings_hooks:
            for event_name, items in settings_hooks.items():
                try:
                    event = HookEvent(event_name)
                except ValueError:
                    continue
                specs = bundle.by_event.setdefault(event, [])
                for item in items or []:
                    if isinstance(item, str):
                        specs.append(HookSpec(event=event, type="template", run=item))
                    elif isinstance(item, dict):
                        specs.append(
                            HookSpec(
                                event=event,
                                matcher=item.get("matcher"),
                                type=item.get("type", "template"),
                                run=item.get("run", ""),
                                timeout_s=int(item.get("timeout_s", 10)),
                            )
                        )
        if agent_hooks:
            for event, specs in agent_hooks.items():
                bundle.by_event.setdefault(event, []).extend(specs)
        return bundle

    def for_event(self, event: HookEvent) -> list[HookSpec]:
        return list(self.by_event.get(event, []))


class HookManager:
    def __init__(self, bundle: HookBundle, *, store=None, run_id: str | None = None) -> None:
        self.bundle = bundle
        self.store = store
        self.run_id = run_id

    def fire(self, event: HookEvent, *, matcher_value: str | None = None, payload: dict[str, Any] | None = None) -> HookResponse:
        payload = payload or {}
        merged = HookResponse()
        rendered_log: list[str] = []
        for spec in self.bundle.for_event(event):
            if spec.matcher and matcher_value is not None:
                if not re.search(spec.matcher, matcher_value):
                    continue
            if spec.type == "template":
                rendered = self._format(spec.run, event=event, **payload)
                rendered_log.append(rendered)
                continue
            response = self._run_command(spec, event=event, payload=payload)
            if response.additional_context:
                merged.additional_context = (
                    (merged.additional_context or "") + "\n" + response.additional_context
                ).strip()
            if response.transform:
                merged.transform = {**(merged.transform or {}), **response.transform}
            if response.block:
                merged.block = True
                merged.reason = response.reason or "blocked by hook"
                self._record(event, payload, rendered_log, blocked=True, reason=merged.reason)
                return merged
        if rendered_log:
            self._record(event, payload, rendered_log, blocked=False)
        return merged

    def _record(self, event: HookEvent, payload: dict[str, Any], rendered: list[str], *, blocked: bool, reason: str | None = None) -> None:
        if self.store is None or self.run_id is None:
            return
        self.store.record(
            self.run_id,
            kind="hook",
            role="hook",
            payload={
                "event": str(event),
                "rendered": rendered,
                "blocked": blocked,
                "reason": reason,
                "payload_keys": sorted(payload.keys()),
            },
        )

    def _run_command(self, spec: HookSpec, *, event: HookEvent, payload: dict[str, Any]) -> HookResponse:
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", spec.run],
                input=json.dumps({"event": str(event), **payload}, default=str),
                capture_output=True,
                text=True,
                timeout=max(1, spec.timeout_s),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return HookResponse(block=True, reason=f"hook timeout ({spec.timeout_s}s)")
        if proc.returncode != 0:
            return HookResponse(block=True, reason=(proc.stderr or proc.stdout or "non-zero exit").strip()[:500])
        out = proc.stdout.strip()
        if not out:
            return HookResponse()
        try:
            return HookResponse.model_validate(json.loads(out))
        except Exception:
            return HookResponse(additional_context=out[:2000])

    @staticmethod
    def _format(template: str, **kwargs: Any) -> str:
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return template
