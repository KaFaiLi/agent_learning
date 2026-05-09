import pytest

from agent_learning.models import ToolCall
from agent_learning.tools import ToolContext, ToolRegistry
from agent_learning.tools._sandbox import SandboxError, resolve_path
from agent_learning.tools.fs import register_fs_tools


def _registry_and_ctx(workspace):
    registry = ToolRegistry()
    register_fs_tools(registry)
    ctx = ToolContext(workspace=workspace)
    return registry, ctx


def test_sandbox_rejects_traversal(tmp_path):
    with pytest.raises(SandboxError):
        resolve_path(tmp_path, "../escape.txt")


def test_write_then_edit_then_read(tmp_path):
    registry, ctx = _registry_and_ctx(tmp_path)
    write = registry.execute(ToolCall(id="1", name="write_file", arguments={"path": "a.txt", "content": "hello world"}), ctx)
    assert write.ok, write.content
    edit = registry.execute(
        ToolCall(id="2", name="edit_file", arguments={"path": "a.txt", "old_string": "world", "new_string": "claude"}),
        ctx,
    )
    assert edit.ok, edit.content
    read = registry.execute(ToolCall(id="3", name="read_file", arguments={"path": "a.txt"}), ctx)
    assert "hello claude" in read.content


def test_edit_file_rejects_ambiguous_match(tmp_path):
    registry, ctx = _registry_and_ctx(tmp_path)
    (tmp_path / "x.txt").write_text("aa aa", encoding="utf-8")
    result = registry.execute(
        ToolCall(id="1", name="edit_file", arguments={"path": "x.txt", "old_string": "aa", "new_string": "b"}),
        ctx,
    )
    assert not result.ok
    assert "matched" in result.content


def test_glob_and_grep(tmp_path):
    registry, ctx = _registry_and_ctx(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("import os\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("import sys\n", encoding="utf-8")

    glob_res = registry.execute(ToolCall(id="1", name="glob", arguments={"pattern": "src/*.py"}), ctx)
    assert "src/a.py" in glob_res.content and "src/b.py" in glob_res.content

    grep_res = registry.execute(ToolCall(id="2", name="grep", arguments={"pattern": "import sys"}), ctx)
    assert "src/b.py:1: import sys" in grep_res.content


def test_write_outside_workspace_blocked(tmp_path):
    registry, ctx = _registry_and_ctx(tmp_path)
    result = registry.execute(
        ToolCall(id="1", name="write_file", arguments={"path": "../escape.txt", "content": "x"}),
        ctx,
    )
    assert not result.ok
    assert "outside" in result.content
