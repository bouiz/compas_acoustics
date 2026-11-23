---
title: "Docstrings render as Markdown (MkDocs + mkdocstrings), and how to check a docstring-only change"
author: Achilleas
date: 2026-11-23
branch: feat/coefficient-lab
cost: { minutes: 1, tokens: 9874 }
touches:
  - docs/_ext/griffe_sphinx_roles.py
  - docs/_ext/mkdocs_hooks.py
  - docs/_tools/capture_grasshopper.py
---

## Decided
- 2026-11-23: the API docs are MkDocs + mkdocstrings (docstring_style: numpy, plus docs/_ext/griffe_sphinx_roles.py for :func: roles). Docstring text renders as Markdown, not RST. The "RST" in the W505 rewrap learning means Markdown. Its pitfalls (a list item's hanging indent, a line starting "0." or "- ") hold just the same.
- To check a docstring-only change, build both commits in throwaway worktrees (mkdocs build -d <dir>) and diff the HTML with whitespace, id and href normalised. For 0ac4c57 (the W505 rewrap) no page differed.

## Tried and ruled out
- `mkdocs build` on feat/coefficient-lab: on_pre_build stops with a BuildError, because the lab components have no Grasshopper screenshots yet. There's no flag to skip it. In a throwaway worktree, make on_pre_build in docs/_ext/mkdocs_hooks.py pass instead of raising PluginError.

## Watch out
- The docs build on this branch fails until the lab's screenshots are captured in Rhino (docs/_tools/capture_grasshopper.py).
