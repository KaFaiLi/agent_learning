from __future__ import annotations

from pathlib import Path

from agent_learning.models import AgentCard
import yaml


class AgentRegistry:
    def __init__(self, agent_dir: Path) -> None:
        self.agent_dir = agent_dir
        self._agents: dict[str, AgentCard] = {}

    def refresh(self) -> None:
        self._agents = {}
        if not self.agent_dir.exists():
            return

        for path in sorted(self.agent_dir.glob("*.md")):
            document = self._parse_document(path)
            self._agents[document.name] = document

    def list_agents(self) -> list[AgentCard]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents.keys())

    def get(self, name: str) -> AgentCard | None:
        return self._agents.get(name)

    def _parse_document(self, path: Path) -> AgentCard:
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        payload = yaml.safe_load(frontmatter) if frontmatter else {}
        payload = payload or {}
        return AgentCard(
            name=payload.get("name", path.stem),
            description=payload.get("description", "No description provided."),
            system_prompt=body.strip(),
            tools=payload.get("tools", []),
            subagents=payload.get("subagents", []),
            skills=payload.get("skills", []),
            hooks=payload.get("hooks", {}),
            memory_policy=payload.get(
                "memory_policy",
                "Summarize important changes and decisions between steps.",
            ),
        )


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text

    _, rest = text.split("---\n", 1)
    frontmatter, body = rest.split("\n---\n", 1)
    return frontmatter, body
