from __future__ import annotations

from pathlib import Path

import yaml

from agent_learning.models import AcceptanceSpec, AgentCard, HookEvent, HookSpec


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file with YAML frontmatter into (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw_meta = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    meta = yaml.safe_load(raw_meta) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, body


def parse_hooks(meta: dict) -> dict[HookEvent, list[HookSpec]]:
    raw = meta.get("hooks") or {}
    out: dict[HookEvent, list[HookSpec]] = {}
    for event_name, items in raw.items():
        try:
            event = HookEvent(event_name)
        except ValueError:
            continue
        specs: list[HookSpec] = []
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
        if specs:
            out[event] = specs
    return out


class AgentRegistry:
    def __init__(self, agent_dir: Path) -> None:
        self.agent_dir = agent_dir
        self._agents: dict[str, AgentCard] = {}
        self.refresh()

    def refresh(self) -> None:
        self._agents = {}
        if not self.agent_dir.exists():
            return
        for path in sorted(self.agent_dir.glob("*.md")):
            card = self._load(path)
            if card is not None:
                self._agents[card.name] = card

    def _load(self, path: Path) -> AgentCard | None:
        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text)
        if not meta.get("name"):
            return None
        acceptance_meta = meta.get("acceptance")
        acceptance = AcceptanceSpec(**acceptance_meta) if isinstance(acceptance_meta, dict) else None
        return AgentCard(
            name=meta["name"],
            description=meta.get("description", ""),
            system_prompt=body.strip(),
            tools=list(meta.get("tools") or ["*"]),
            subagents=list(meta.get("subagents") or []),
            skills=list(meta.get("skills") or []),
            hooks=parse_hooks(meta),
            bash_allow=list(meta.get("bash_allow") or []),
            bash_deny=list(meta.get("bash_deny") or []),
            acceptance=acceptance,
            max_iterations=int(meta.get("max_iterations", 12)),
        )

    def get(self, name: str) -> AgentCard | None:
        return self._agents.get(name)

    def list(self) -> list[AgentCard]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return sorted(self._agents.keys())
