from __future__ import annotations

import re
import shlex
from pathlib import Path

DEFAULT_DENY_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\brm\s+--no-preserve-root\b",
    r"\bsudo\b",
    r"\bsu\b",
    r"\bcurl\s+-[a-zA-Z]*\s*https?://",
    r"\bwget\s+https?://",
    r"\bssh\b",
    r"\bscp\b",
    r"\bnc\b",
    r"\bmkfs\b",
    r"\bdd\s+if=.*of=/",
    r":\(\)\s*\{",  # fork bomb
    r">\s*/dev/sd",
    r"\bchmod\s+777\b",
    r"\bgit\s+push\b.*--force",
]


class SandboxError(RuntimeError):
    pass


def resolve_path(workspace: Path, path: str) -> Path:
    """Resolve `path` inside `workspace`. Reject traversal and absolute paths outside it."""
    if not path:
        raise SandboxError("Path is required.")
    raw = Path(path)
    candidate = (workspace / raw).resolve() if not raw.is_absolute() else raw.resolve()
    workspace = workspace.resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise SandboxError(f"Path {path!r} is outside the workspace {workspace}.") from exc
    return candidate


def screen_command(command: str, *, extra_allow: list[str] = (), extra_deny: list[str] = ()) -> None:
    """Raise SandboxError if `command` matches a denied pattern (after allow overrides)."""
    if not command.strip():
        raise SandboxError("Empty command.")
    for pattern in extra_allow:
        if re.search(pattern, command):
            return
    for pattern in list(DEFAULT_DENY_PATTERNS) + list(extra_deny):
        if re.search(pattern, command):
            raise SandboxError(f"Command blocked by safety policy: matches {pattern!r}.")
    # Block writing into common system dirs even without rm.
    try:
        tokens = shlex.split(command)
    except ValueError:
        return
    for token in tokens:
        if token.startswith(("/etc/", "/usr/", "/bin/", "/sbin/", "/boot/", "/sys/", "/proc/")):
            raise SandboxError(f"Command touches protected path: {token}.")
