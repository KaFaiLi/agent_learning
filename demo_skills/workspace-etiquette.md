---
name: workspace-etiquette
description: Always-on guidance about safe edits inside the workspace.
---
- Stay inside the configured workspace; tools refuse paths outside it.
- Prefer edit_file over write_file when the file already exists.
- Read a file before editing it; if old_string isn't unique, add more context.
- Bash is sandboxed: rm -rf, sudo, outbound curl/wget/ssh are blocked.
- Use todo_write for multi-step plans and update statuses as you go.
