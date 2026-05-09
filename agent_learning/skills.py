from __future__ import annotations

from pathlib import Path

from agent_learning.models import SkillCard
import yaml


class SkillRegistry:
    def __init__(self, skill_dir: Path) -> None:
        self.skill_dir = skill_dir
        self._skills: dict[str, SkillCard] = {}

    def refresh(self) -> None:
        self._skills = {}
        if not self.skill_dir.exists():
            return

        for path in sorted(self.skill_dir.glob("*.md")):
            self._skills[path.stem] = self._parse_document(path)

    def list_skills(self) -> list[SkillCard]:
        return list(self._skills.values())

    def get(self, name: str) -> SkillCard | None:
        return self._skills.get(name)

    def _parse_document(self, path: Path) -> SkillCard:
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        payload = yaml.safe_load(frontmatter) if frontmatter else {}
        payload = payload or {}
        return SkillCard(
            name=payload.get("name", path.stem),
            description=payload.get("description", "No description provided."),
            instructions=body.strip(),
        )


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text

    _, rest = text.split("---\n", 1)
    frontmatter, body = rest.split("\n---\n", 1)
    return frontmatter, body
