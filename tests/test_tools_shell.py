from agent_learning.models import ToolCall
from agent_learning.tools import ToolContext, ToolRegistry
from agent_learning.tools.shell import register_shell_tools


def _setup(tmp_path):
    registry = ToolRegistry()
    register_shell_tools(registry)
    ctx = ToolContext(workspace=tmp_path)
    return registry, ctx


def test_bash_runs_simple_command(tmp_path):
    registry, ctx = _setup(tmp_path)
    result = registry.execute(ToolCall(id="1", name="bash", arguments={"command": "echo hello"}), ctx)
    assert result.ok, result.content
    assert "hello" in result.content


def test_bash_blocks_rm_rf(tmp_path):
    registry, ctx = _setup(tmp_path)
    result = registry.execute(ToolCall(id="1", name="bash", arguments={"command": "rm -rf /"}), ctx)
    assert not result.ok
    assert "blocked" in result.content.lower()


def test_bash_blocks_sudo(tmp_path):
    registry, ctx = _setup(tmp_path)
    result = registry.execute(ToolCall(id="1", name="bash", arguments={"command": "sudo ls"}), ctx)
    assert not result.ok


def test_bash_reports_nonzero_exit(tmp_path):
    registry, ctx = _setup(tmp_path)
    result = registry.execute(ToolCall(id="1", name="bash", arguments={"command": "exit 7"}), ctx)
    assert not result.ok
    assert "exit=7" in result.content
