---
name: bootstrap-runtime
description: Scaffold or extend the plain-LLM runtime without introducing an agentic framework.
---
Focus on small, composable runtime slices.

Prefer separate modules for:
- Azure OpenAI client logic
- tool execution
- MCP adapters
- memory persistence
- usage and cost accounting
- TUI rendering
