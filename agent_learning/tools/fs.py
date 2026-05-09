from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from agent_learning.models import ToolResult
from agent_learning.tools import Tool, ToolContext
from agent_learning.tools._sandbox import SandboxError, resolve_path

_MAX_BYTES = 200_000


def _ok(name: str, content: str) -> ToolResult:
    return ToolResult(tool_call_id="", name=name, content=content, ok=True)


def _err(name: str, content: str) -> ToolResult:
    return ToolResult(tool_call_id="", name=name, content=content, ok=False)


class ReadFileArgs(BaseModel):
    path: str = Field(description="Workspace-relative path to read.")


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace. Returns up to 200KB."
    args_model = ReadFileArgs

    def run(self, args: ReadFileArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = resolve_path(ctx.workspace, args.path)
        except SandboxError as exc:
            return _err(self.name, str(exc))
        if not path.exists():
            return _err(self.name, f"File not found: {args.path}")
        if path.is_dir():
            return _err(self.name, f"{args.path} is a directory.")
        data = path.read_bytes()[:_MAX_BYTES]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return _err(self.name, f"{args.path} is not UTF-8 text.")
        truncated = "\n... (truncated)" if path.stat().st_size > _MAX_BYTES else ""
        return _ok(self.name, text + truncated)


class WriteFileArgs(BaseModel):
    path: str
    content: str
    overwrite: bool = Field(default=False, description="Set true to overwrite an existing file.")


class WriteFileTool:
    name = "write_file"
    description = "Create or overwrite a file in the workspace. Refuses to overwrite unless overwrite=true."
    args_model = WriteFileArgs

    def run(self, args: WriteFileArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = resolve_path(ctx.workspace, args.path)
        except SandboxError as exc:
            return _err(self.name, str(exc))
        if path.exists() and not args.overwrite:
            return _err(self.name, f"{args.path} exists. Pass overwrite=true to replace it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.content, encoding="utf-8")
        return _ok(self.name, f"Wrote {len(args.content)} chars to {args.path}.")


class EditFileArgs(BaseModel):
    path: str
    old_string: str
    new_string: str
    replace_all: bool = False


class EditFileTool:
    name = "edit_file"
    description = "Exact-match string replacement in a file. old_string must appear once unless replace_all=true."
    args_model = EditFileArgs

    def run(self, args: EditFileArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = resolve_path(ctx.workspace, args.path)
        except SandboxError as exc:
            return _err(self.name, str(exc))
        if not path.exists():
            return _err(self.name, f"File not found: {args.path}")
        text = path.read_text(encoding="utf-8")
        if args.old_string == "":
            return _err(self.name, "old_string must be non-empty.")
        count = text.count(args.old_string)
        if count == 0:
            return _err(self.name, "old_string not found.")
        if count > 1 and not args.replace_all:
            return _err(self.name, f"old_string matched {count} times. Pass replace_all=true or add more context.")
        new_text = text.replace(args.old_string, args.new_string) if args.replace_all else text.replace(args.old_string, args.new_string, 1)
        path.write_text(new_text, encoding="utf-8")
        return _ok(self.name, f"Replaced {count if args.replace_all else 1} occurrence(s) in {args.path}.")


class ListDirArgs(BaseModel):
    path: str = "."


class ListDirTool:
    name = "list_dir"
    description = "List entries in a workspace directory."
    args_model = ListDirArgs

    def run(self, args: ListDirArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = resolve_path(ctx.workspace, args.path)
        except SandboxError as exc:
            return _err(self.name, str(exc))
        if not path.exists():
            return _err(self.name, f"Not found: {args.path}")
        if not path.is_dir():
            return _err(self.name, f"{args.path} is not a directory.")
        entries = []
        for child in sorted(path.iterdir()):
            kind = "d" if child.is_dir() else "f"
            entries.append(f"{kind} {child.name}")
        return _ok(self.name, "\n".join(entries) or "(empty)")


class GlobArgs(BaseModel):
    pattern: str = Field(description="Glob like 'src/**/*.py'. Workspace-rooted.")
    limit: int = 200


class GlobTool:
    name = "glob"
    description = "Find files matching a glob pattern under the workspace."
    args_model = GlobArgs

    def run(self, args: GlobArgs, ctx: ToolContext) -> ToolResult:
        results: list[str] = []
        try:
            for path in ctx.workspace.glob(args.pattern):
                rel = path.relative_to(ctx.workspace)
                results.append(str(rel))
                if len(results) >= args.limit:
                    break
        except (ValueError, OSError) as exc:
            return _err(self.name, f"Glob error: {exc}")
        return _ok(self.name, "\n".join(sorted(results)) or "(no matches)")


class GrepArgs(BaseModel):
    pattern: str
    path: str | None = Field(default=None, description="Optional subdirectory or file. Defaults to workspace root.")
    regex: bool = False
    limit: int = 100


class GrepTool:
    name = "grep"
    description = "Search files for a string or regex. Returns matching lines as 'path:lineno: text'."
    args_model = GrepArgs

    def run(self, args: GrepArgs, ctx: ToolContext) -> ToolResult:
        try:
            root = resolve_path(ctx.workspace, args.path) if args.path else ctx.workspace
        except SandboxError as exc:
            return _err(self.name, str(exc))
        try:
            matcher = re.compile(args.pattern) if args.regex else None
        except re.error as exc:
            return _err(self.name, f"Bad regex: {exc}")
        files: list[Path] = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        out: list[str] = []
        for path in files:
            if any(part.startswith(".") and part not in (".", "..") for part in path.relative_to(ctx.workspace).parts):
                continue  # skip dot-dirs (.git, .agent_runtime, etc.)
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                hit = matcher.search(line) if matcher else (args.pattern in line)
                if hit:
                    rel = path.relative_to(ctx.workspace)
                    out.append(f"{rel}:{i}: {line}")
                    if len(out) >= args.limit:
                        break
            if len(out) >= args.limit:
                break
        return _ok(self.name, "\n".join(out) or "(no matches)")


def register_fs_tools(registry) -> None:
    for tool in (ReadFileTool(), WriteFileTool(), EditFileTool(), ListDirTool(), GlobTool(), GrepTool()):
        registry.register(tool)
