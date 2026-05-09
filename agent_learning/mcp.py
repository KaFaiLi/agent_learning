from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    transport: str = "stdio"


class MCPBridge:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._servers: list[MCPServerConfig] = []
        self._load_error: str | None = None
        self.refresh()

    def refresh(self) -> None:
        self._servers = []
        self._load_error = None
        if not self.config_path.exists():
            return

        try:
            payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            self._load_error = f"Invalid MCP config in {self.config_path.name}: {exc}"
            return

        for item in payload.get("servers", []):
            self._servers.append(
                MCPServerConfig(
                    name=item["name"],
                    command=item["command"],
                    args=item.get("args", []),
                    transport=item.get("transport", "stdio"),
                )
            )

    def list_servers(self) -> list[MCPServerConfig]:
        return list(self._servers)

    def describe(self) -> str:
        if self._load_error:
            return self._load_error
        if not self._servers:
            return "No MCP servers configured. Add mcp_servers.yaml to register stdio servers."
        return "\n".join(
            f"- {server.name}: {server.command} {' '.join(server.args)} ({server.transport})".strip()
            for server in self._servers
        )
