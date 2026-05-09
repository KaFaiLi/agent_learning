---
name: budget-review
description: Inspect or improve token usage tracking, pricing estimates, and memory summaries.
---
Whenever you touch the loop or LLM call path:
- keep per-step usage snapshots
- preserve cumulative run totals
- surface estimated cost in the UI
- summarize important memory updates instead of dumping raw transcripts
