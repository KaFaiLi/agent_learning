import sys
import textwrap

import yaml

from agent_learning.mcp import MCPBridge


FAKE_SERVER = textwrap.dedent(
    """
    import json, sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method")
        rpc_id = msg.get("id")
        if method == "initialize":
            response = {"jsonrpc": "2.0", "id": rpc_id, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "0.0"}}}
            sys.stdout.write(json.dumps(response) + "\\n")
            sys.stdout.flush()
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            response = {"jsonrpc": "2.0", "id": rpc_id, "result": {"tools": [{"name": "echo", "description": "echo back", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}}
            sys.stdout.write(json.dumps(response) + "\\n")
            sys.stdout.flush()
        elif method == "tools/call":
            text = msg["params"]["arguments"].get("text", "")
            response = {"jsonrpc": "2.0", "id": rpc_id, "result": {"content": [{"type": "text", "text": "echoed:" + text}]}}
            sys.stdout.write(json.dumps(response) + "\\n")
            sys.stdout.flush()
    """
).strip()


def test_mcp_bridge_handshake_and_call(tmp_path):
    server_script = tmp_path / "fake_server.py"
    server_script.write_text(FAKE_SERVER, encoding="utf-8")
    config_path = tmp_path / "mcp_servers.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "servers": [
                    {"name": "fake", "command": sys.executable, "args": ["-u", str(server_script)]}
                ]
            }
        ),
        encoding="utf-8",
    )
    bridge = MCPBridge(config_path)
    try:
        tools = bridge.list_tools()
        names = {t["name"] for t in tools}
        assert "echo" in names
        result = bridge.call("fake", "echo", {"text": "hi"})
        assert result.ok
        assert "echoed:hi" in result.content
    finally:
        bridge.shutdown()
