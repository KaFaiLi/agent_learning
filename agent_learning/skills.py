from __future__ import annotations

from pathlib import Path

from agent_learning.agents import split_frontmatter
from agent_learning.models import SkillCard


class SkillRegistry:
    def __init__(self, skill_dir: Path) -> None:
        self.skill_dir = skill_dir
        self._skills: dict[str, SkillCard] = {}
        self.refresh()

    def refresh(self) -> None:
        self._skills = {}
        if not self.skill_dir.exists():
            return
        for path in sorted(self.skill_dir.glob("*.md")):
            card = self._load(path)
            if card is not None:
                self._skills[card.name] = card

    def _load(self, path: Path) -> SkillCard | None:
        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text)
        if not meta.get("name"):
            return None
        return SkillCard(
            name=meta["name"],
            description=meta.get("description", ""),
            triggers=list(meta.get("triggers") or []),
            body=body.strip(),
        )

    def get(self, name: str) -> SkillCard | None:
        return self._skills.get(name)

    def list(self) -> list[SkillCard]:
        return list(self._skills.values())

    def select(self, *, declared: list[str], goal: str, recent_text: str = "") -> list[SkillCard]:
        """Return declared skills plus any whose triggers match goal/recent text."""
        haystack = (goal + "\n" + recent_text).lower()
        chosen: dict[str, SkillCard] = {}
        for name in declared:
            card = self.get(name)
            if card is not None:
                chosen[card.name] = card
        for card in self._skills.values():
            if card.name in chosen:
                continue
            if any(trigger.lower() in haystack for trigger in card.triggers if trigger):
                chosen[card.name] = card
        return list(chosen.values())

    def render(self, cards: list[SkillCard], *, max_chars: int = 4000) -> str:
        if not cards:
            return ""
        sections: list[str] = []
        budget = max_chars
        for card in cards:
            block = f"### {card.name}\n{card.description}\n\n{card.body}".strip()
            if len(block) > budget:
                block = block[: max(budget - 50, 0)] + "\n... (truncated)"
            sections.append(block)
            budget -= len(block) + 2
            if budget <= 0:
                break
        return "## Skills\n\n" + "\n\n".join(sections)
