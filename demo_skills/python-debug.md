---
name: python-debug
description: Heuristics for fixing failing Python tests.
triggers:
  - python
  - pytest
  - traceback
  - typeerror
  - importerror
---
- Run `pytest -q` first to see the failure shape, then `pytest -x -q <node>` to focus on one test.
- Read the traceback bottom-up; the deepest frame in user code is usually the bug.
- Reproduce locally with `python -c` for quick checks before editing.
- Don't widen except clauses; fix the underlying cause.
- Re-run the suite after every edit; stop when green.
