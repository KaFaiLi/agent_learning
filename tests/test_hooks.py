import os
import stat

from agent_learning.hooks import HookBundle, HookManager
from agent_learning.models import HookEvent, HookSpec


def test_template_hook_renders_payload():
    bundle = HookBundle(by_event={HookEvent.PRE_TOOL_USE: [HookSpec(event=HookEvent.PRE_TOOL_USE, type="template", run="hello {tool}")]})
    manager = HookManager(bundle)
    response = manager.fire(HookEvent.PRE_TOOL_USE, payload={"tool": "bash"})
    assert response.block is False


def test_command_hook_blocks_on_nonzero_exit(tmp_path):
    script = tmp_path / "deny.sh"
    script.write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
    bundle = HookBundle(by_event={HookEvent.PRE_TOOL_USE: [HookSpec(event=HookEvent.PRE_TOOL_USE, type="command", run=str(script), timeout_s=5)]})
    manager = HookManager(bundle)
    response = manager.fire(HookEvent.PRE_TOOL_USE, payload={"tool": "bash"})
    assert response.block is True


def test_command_hook_transforms_via_stdout(tmp_path):
    script = tmp_path / "transform.sh"
    script.write_text(
        '#!/bin/bash\necho \'{"transform":{"path":"safe.txt"}}\'\n',
        encoding="utf-8",
    )
    os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
    bundle = HookBundle(by_event={HookEvent.PRE_TOOL_USE: [HookSpec(event=HookEvent.PRE_TOOL_USE, type="command", run=str(script), timeout_s=5)]})
    manager = HookManager(bundle)
    response = manager.fire(HookEvent.PRE_TOOL_USE, payload={"tool": "write_file"})
    assert response.block is False
    assert response.transform == {"path": "safe.txt"}


def test_matcher_filters_hooks():
    bundle = HookBundle(
        by_event={
            HookEvent.PRE_TOOL_USE: [
                HookSpec(event=HookEvent.PRE_TOOL_USE, matcher="^bash$", type="template", run="bash only {tool}"),
            ]
        }
    )
    manager = HookManager(bundle)
    # should not raise; matcher rejects
    response = manager.fire(HookEvent.PRE_TOOL_USE, matcher_value="read_file", payload={"tool": "read_file"})
    assert response.block is False
