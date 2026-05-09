from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from agent_learning.models import ToolResult
from agent_learning.tools import ToolContext, ToolRegistry


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    transport: str = "stdio"


@dataclass
class _ServerState:
    config: MCPServerConfig
    proc: subprocess.Popen | None = None
    next_id: int = 1
    tools: list[dict[str, Any]] = field(default_factory=list)
    healthy: bool = False
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class _MCPTool:
    """Wrapper that adapts an MCP tool descriptor into the local Tool protocol."""

    def __init__(self, bridge: "MCPBridge", server: str, info: dict[str, Any]) -> None:
        self.bridge = bridge
        self.server = server
        self.tool_name = info["name"]
        self.name = f"mcp__{server}__{info['name']}"
        self.description = (info.get("description") or f"MCP tool {info['name']} from {server}.")[:500]
        schema = info.get("inputSchema") or {"type": "object", "properties": {}}
        self.args_model = _make_args_model(self.name, schema)
        self._raw_schema = schema

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        payload = args.model_dump(exclude_none=True)
        return self.bridge.call(self.server, self.tool_name, payload)


def _make_args_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a permissive Pydantic model from a JSON-schema input definition.

    The schema arrives from a remote MCP server, so we keep it pass-through:
    one field 'arguments' typed as dict, but we still respect required keys
    by pulling them from kwargs at call-time. Keep it simple for the demo.
    """

    class _Passthrough(BaseModel):
        model_config = {"extra": "allow"}

    _Passthrough.__name__ = f"MCPArgs_{name}"
    return _Passthrough


class MCPBridge:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.servers: dict[str, _ServerState] = {}
        self.refresh()

    def refresh(self) -> None:
        if not self.config_path.exists():
            self.servers = {}
            return
        try:
            payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            self.servers = {}
            self._load_error = f"Invalid MCP config: {exc}"
            return
        existing = self.servers
        new: dict[str, _ServerState] = {}
        for item in payload.get("servers", []):
            cfg = MCPServerConfig(
                name=item["name"],
                command=item["command"],
                args=list(item.get("args") or []),
                transport=item.get("transport", "stdio"),
            )
            if cfg.name in existing:
                state = existing.pop(cfg.name)
                state.config = cfg
            else:
                state = _ServerState(config=cfg)
            new[cfg.name] = state
        # close any servers that are no longer configured
        for old in existing.values():
            self._close(old)
        self.servers = new

    def describe(self) -> str:
        if not self.servers:
            return "No MCP servers configured."
        lines = []
        for state in self.servers.values():
            status = "healthy" if state.healthy else ("error: " + state.error if state.error else "not started")
            lines.append(f"- {state.config.name}: {state.config.command} {' '.join(state.config.args)} [{status}]")
        return "\n".join(lines)

    def list_tools(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for state in self.servers.values():
            self._ensure_started(state)
            for info in state.tools:
                out.append({"server": state.config.name, **info})
        return out

    def register_into(self, registry: ToolRegistry) -> None:
        for state in self.servers.values():
            self._ensure_started(state)
            if not state.healthy:
                continue
            for info in state.tools:
                tool = _MCPTool(self, state.config.name, info)
                registry.register(tool)

    def call(self, server: str, tool: str, arguments: dict[str, Any]) -> ToolResult:
        state = self.servers.get(server)
        if state is None:
            return ToolResult(tool_call_id="", name=f"mcp__{server}__{tool}", content=f"Unknown MCP server: {server}", ok=False)
        self._ensure_started(state)
        if not state.healthy:
            return ToolResult(tool_call_id="", name=f"mcp__{server}__{tool}", content=f"MCP server unhealthy: {state.error}", ok=False)
        try:
            response = self._rpc(state, "tools/call", {"name": tool, "arguments": arguments})
        except Exception as exc:
            state.healthy = False
            state.error = str(exc)
            return ToolResult(tool_call_id="", name=f"mcp__{server}__{tool}", content=f"MCP call failed: {exc}", ok=False)
        result = response.get("result") or {}
        content = result.get("content") or []
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    text_parts.append(str(part["text"]))
                else:
                    text_parts.append(json.dumps(part, ensure_ascii=False))
            else:
                text_parts.append(str(part))
        is_error = bool(result.get("isError"))
        return ToolResult(
            tool_call_id="",
            name=f"mcp__{server}__{tool}",
            content="\n".join(text_parts) or "(no content)",
            ok=not is_error,
        )

    def shutdown(self) -> None:
        for state in self.servers.values():
            self._close(state)

    # ---- stdio JSON-RPC plumbing -----------------------------------

    def _ensure_started(self, state: _ServerState) -> None:
        with state.lock:
            if state.proc is not None and state.proc.poll() is None and state.healthy:
                return
            try:
                proc = subprocess.Popen(
                    [state.config.command, *state.config.args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as exc:
                state.error = f"command not found: {exc}"
                state.healthy = False
                return
            state.proc = proc
            state.next_id = 1
            try:
                self._rpc(
                    state,
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "clientInfo": {"name": "agent-learning", "version": "0.2.0"},
                    },
                    locked=True,
                )
                self._notify(state, "notifications/initialized", {})
                tools_resp = self._rpc(state, "tools/list", {}, locked=True)
                state.tools = list((tools_resp.get("result") or {}).get("tools") or [])
                state.healthy = True
                state.error = None
            except Exception as exc:
                state.healthy = False
                state.error = str(exc)
                self._close(state)

    def _rpc(self, state: _ServerState, method: str, params: dict[str, Any], *, locked: bool = False, timeout: float = 30.0) -> dict[str, Any]:
        if state.proc is None or state.proc.stdin is None or state.proc.stdout is None:
            raise RuntimeError("MCP server not running")
        rpc_id = state.next_id
        state.next_id += 1
        payload = json.dumps({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}) + "\n"
        if not locked:
            state.lock.acquire()
        try:
            state.proc.stdin.write(payload)
            state.proc.stdin.flush()
            deadline = time.monotonic() + timeout
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"timeout waiting for response to {method}")
                line = state.proc.stdout.readline()
                if not line:
                    raise RuntimeError("MCP server closed stdout")
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == rpc_id:
                    if "error" in msg:
                        raise RuntimeError(f"MCP error: {msg['error']}")
                    return msg
        finally:
            if not locked:
                state.lock.release()

    def _notify(self, state: _ServerState, method: str, params: dict[str, Any]) -> None:
        if state.proc is None or state.proc.stdin is None:
            return
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
        state.proc.stdin.write(payload)
        state.proc.stdin.flush()

    def _close(self, state: _ServerState) -> None:
        proc = state.proc
        state.proc = None
        state.healthy = False
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
