# agent-learning

A small Claude-Code-style TUI agent demo. Azure OpenAI under the hood, with
multi-tool-per-turn function calling, sandboxed file/bash tools, lifecycle
hooks, markdown-defined agents and skills, sub-agents, an MCP stdio bridge,
and an optional Ralph loop that iterates until a goal is verified — all in
about 1.6k lines and four runtime dependencies (`openai`, `pydantic`, `PyYAML`,
`textual`).

## Why this exists

It's a learning demo that mirrors the moving parts of Claude Code without a
framework, so you can read the whole thing end-to-end:

- Native OpenAI tool-calling loop in `engine.py`
- File / bash / glob / grep tools constrained to a workspace in `tools/`
- Hook system with `SessionStart`, `UserPromptSubmit`, `PreToolUse`,
  `PostToolUse`, `AgentStart`, `AgentStop`, `Stop` events that can `block`,
  `transform`, or inject extra context — see `hooks.py`
- Markdown agents in `demo_agents/` with YAML frontmatter (tools, sub-agents,
  skills, hooks, bash allow/deny, acceptance criteria)
- Markdown skills in `demo_skills/` with `triggers:` for keyword auto-surfacing
- MCP stdio bridge in `mcp.py` (newline-delimited JSON-RPC, no extra deps)
- Ralph loop in `ralph.py` (LLM rubric and/or `verify_command` exit code)
- Textual TUI in `tui/app.py` with slash commands, streaming events, budget bar
- Append-only JSONL run log in `.agent_runtime/runs/`

## Install

```
uv sync
cp .env.example .env             # fill in Azure OpenAI values
cp settings.example.json settings.json   # optional global hooks/allowlists
cp mcp_servers.example.yaml mcp_servers.yaml  # optional, needs npx for filesystem server
```

## Run

```
uv run agent-learning --print-config            # dump resolved settings
uv run agent-learning                           # launch TUI
uv run agent-learning --no-tui --goal "..."     # headless run
uv run agent-learning --no-tui --ralph --agent coder --goal "..."   # Ralph loop
```

In the TUI, type a goal and press Enter, or use slash commands — `/help`,
`/goal`, `/agent <name>`, `/ralph on|off`, `/agents`, `/skills`, `/mcp`,
`/budget`, `/clear`, `/quit`. `Ctrl+R` re-runs the current input, `Ctrl+L`
refreshes registries, `Ctrl+Q` quits.

## Built-in tools

| tool | purpose |
| --- | --- |
| `read_file` / `write_file` / `edit_file` | sandboxed file ops; `edit_file` is exact-match |
| `list_dir` / `glob` / `grep` | workspace inspection |
| `bash` | shell with deny-list (rm -rf, sudo, outbound curl/wget/ssh, ...) |
| `list_agents` / `list_skills` / `mcp_status` / `memory_snapshot` | introspection |
| `todo_write` / `todo_read` | per-run todo list the agent can use |
| `mcp__<server>__<name>` | every tool exposed by configured MCP servers |

Sub-agents declared in an agent's frontmatter appear to the model as virtual
tools whose arguments are `{task, context}`.

## Hooks

Configured in `settings.json` (global) and/or in agent frontmatter under
`hooks:`. Two flavours:

- `template` — a string formatted with the event payload and appended to the
  run log. Never blocks.
- `command` — a shell command receiving the event payload as JSON on stdin.
  Stdout is parsed as `{block, reason, transform, additional_context}`.
  Non-zero exit (or timeout) blocks the action.

`PreToolUse` hooks can return `transform: {<args>}` to mutate the tool call;
`PostToolUse` hooks can return `transform: {content: "..."}` to rewrite the
tool result.

## Skills

Two ways a skill body lands in the system prompt:

1. **Declared** — listed in the agent's frontmatter `skills:`.
2. **Triggered** — the skill's `triggers:` keyword list matches the goal text
   (case-insensitive).

The combined block is appended under `## Skills` and capped at ~4000 chars.

## Ralph loop

Set `acceptance:` on an agent (`verify_command`, `rubric`, or both). With
`--ralph` (or the TUI's `/ralph on`), the engine reruns the goal with the
unresolved gaps until acceptance passes or the budget is exhausted. If both
are set, the shell command is authoritative; the rubric only runs when the
command passes.

## Sub-agents

A sub-agent is just another markdown agent. The parent agent declares it under
`subagents:`. The engine exposes each sub-agent as a virtual function tool;
calls cascade onto the parent's `BudgetState`, depth is capped at 1.

## MCP

Each entry in `mcp_servers.yaml` is launched as a subprocess on first use. We
do an `initialize` handshake, send `notifications/initialized`, then call
`tools/list` and register each tool as `mcp__<server>__<tool>`. Calls go
through `tools/call` and the JSON-RPC response's `content` is flattened to
text for the model.

## Tests

```
uv run pytest -q
```

Tests cover models, store, tool sandbox, hooks, MCP framing (against a fake
stdio server), engine loop (with a stub LLM), and Ralph (both shell-verify and
stubbed-rubric paths). No live Azure call is made.
