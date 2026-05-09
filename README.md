# Agent Learning

This repo now contains the first runnable slice of an agentic LLM demo built without an agent framework. The current implementation focuses on a plain Azure OpenAI planning loop, markdown-defined agents, skill loading, hook execution, memory persistence, usage and cost tracking, and a Textual-based TUI shell.

## Current Features

- Azure OpenAI configuration from local environment variables only
- Root `.env` file is auto-loaded at startup
- Structured loop decisions via typed Pydantic models
- Parquet-backed runtime memory with working and summary entries
- Token and estimated cost tracking with configurable pricing
- Markdown-defined agents with subagent delegation
- Markdown-defined skills that can be surfaced to the runtime
- Lifecycle hooks declared in agent markdown frontmatter
- Tool registry with built-in demo tools
- MCP configuration discovery through a local YAML file
- Console mode and Textual TUI entrypoints

## Environment Variables

Set the Azure settings in your local shell or in a project-root `.env` file before running the app:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION` (optional, defaults to `2024-10-21`)

Optional pricing inputs for estimated cost tracking:

- `AGENT_LEARNING_MODEL_NAME`
- `AGENT_LEARNING_INPUT_PRICE_PER_1K_TOKENS`
- `AGENT_LEARNING_OUTPUT_PRICE_PER_1K_TOKENS`

Optional runtime overrides:

- `AGENT_LEARNING_AGENT_DIR`
- `AGENT_LEARNING_SKILL_DIR`
- `AGENT_LEARNING_MCP_CONFIG`
- `AGENT_LEARNING_MEMORY_PATH`
- `AGENT_LEARNING_MEMORY_DB` (legacy alias; `.db` values are normalized to `.parquet`)
- `AGENT_LEARNING_MAX_ITERATIONS`
- `AGENT_LEARNING_STEP_BUDGET_USD`
- `AGENT_LEARNING_RUN_BUDGET_USD`

## Install

```bash
python -m pip install -e .
```

## Run

Console mode:

```bash
python main.py --once --goal "Inspect the agent markdown, loaded skills, and memory state"
```

TUI mode:

```bash
python main.py
```

Print resolved config:

```bash
python main.py --print-config
```

## MCP Config

The runtime currently discovers MCP server definitions from `mcp_servers.yaml`. Start by copying the sample file and adjusting the command for your local server.

## Project Layout

- `agent_learning/config.py`: environment-backed runtime settings
- `agent_learning/memory.py`: parquet-backed memory and usage persistence
- `agent_learning/llm.py`: Azure OpenAI planner plus heuristic fallback
- `agent_learning/engine.py`: Ralph-loop style orchestration
- `agent_learning/tui/app.py`: Textual UI shell
- `demo_agents/`: markdown-defined agents with hooks and subagents
- `demo_skills/`: markdown-defined runtime skills

## Next Implementation Slices

1. Replace MCP discovery-only scaffolding with live stdio MCP tool bridging.
2. Add richer tool definitions and goal-specific subagent behaviors.
3. Surface event history, skills, and budget state more richly inside the TUI.
4. Add tests for the markdown loaders, hook execution, and runtime loop.

