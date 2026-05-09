from __future__ import annotations

import subprocess

from pydantic import BaseModel, Field

from agent_learning.models import ToolResult
from agent_learning.tools import Tool, ToolContext
from agent_learning.tools._sandbox import SandboxError, screen_command


class BashArgs(BaseModel):
    command: str = Field(description="Shell command to execute in the workspace via /bin/bash -lc.")
    timeout_s: int = 30


class BashTool:
    name = "bash"
    description = "Run a shell command in the workspace. A safety policy denies destructive/network commands."
    args_model = BashArgs

    def run(self, args: BashArgs, ctx: ToolContext) -> ToolResult:
        try:
            screen_command(args.command, extra_allow=ctx.bash_allow, extra_deny=ctx.bash_deny)
        except SandboxError as exc:
            return ToolResult(tool_call_id="", name=self.name, content=str(exc), ok=False)
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", args.command],
                cwd=ctx.workspace,
                capture_output=True,
                text=True,
                timeout=max(1, min(args.timeout_s, 300)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_call_id="",
                name=self.name,
                content=f"Command timed out after {args.timeout_s}s.",
                ok=False,
            )
        out = (proc.stdout or "")[-8000:]
        err = (proc.stderr or "")[-4000:]
        body_parts = [f"exit={proc.returncode}"]
        if out:
            body_parts.append("stdout:\n" + out)
        if err:
            body_parts.append("stderr:\n" + err)
        return ToolResult(
            tool_call_id="",
            name=self.name,
            content="\n".join(body_parts),
            ok=(proc.returncode == 0),
        )


def register_shell_tools(registry) -> None:
    registry.register(BashTool())
