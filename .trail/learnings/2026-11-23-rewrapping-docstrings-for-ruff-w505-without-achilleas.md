---
title: Rewrapping docstrings for ruff W505 without breaking RST
author: Achilleas
date: 2026-11-23
branch: feat/coefficient-lab
cost: { minutes: 4, tokens: 29198 }
touches:
  - src/compas_acoustics/lab/bands.py
  - src/compas_acoustics/lab/geometric.py
  - src/compas_acoustics/lab/grids.py
  - src/compas_acoustics/lab/raycast.py
  - src/compas_acoustics/lab/results.py
  - src/compas_acoustics/lab/sample.py
  - src/compas_acoustics/lab/scattering.py
  - src/compas_acoustics/lab/solvers/__init__.py
  - src/compas_acoustics/lab/solvers/_numcalc.py
  - src/compas_acoustics/lab/solvers/base.py
---

## Decided
- 2026-11-23, feat/coefficient-lab 0ac4c57: the lab's 371 W505 lines (doc lines over 120 columns, main's rule) were rewrapped by a script. Each flagged line and the plain-text lines of its paragraph below it were refilled with textwrap (break_long_words=False, break_on_hyphens=False). Lines above the flagged one were left alone.
- The check: with whitespace and comment markers collapsed, every file's words are the same before and after.

## Tried and ruled out
- Plain textwrap at the paragraph's own indent: it pulled a list item's continuation back to the item's indent (RST needs the hanging indent: 6 for "- ", 7 for "2. "), it broke "whose σ is 0." so that a line began "0. Returns" (an RST numbered list), and it split "# -- planning ----" rules into two lines. The word check misses all three, because only whitespace moves.

## Watch out
- After a rewrap, grep the added lines for starts like `- `, `* `, `\d+. ` and `..`, and for rules split across lines. A word diff doesn't catch these.
