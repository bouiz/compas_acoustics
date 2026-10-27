---
title: "Scattering Lab cache_dir: where BEM runs are kept and how resume works"
author: Achilleas
date: 2026-10-25
branch: feat/coefficient-lab
cost: { minutes: 1, tokens: 31579 }
touches:
  - src/compas_acoustics/lab/solvers/bem.py
---

## Decided
- Leaving `cache_dir` empty does not discard results. BemSolver falls back to `~/.compas_acoustics/lab_cache` (src/compas_acoustics/lab/solvers/bem.py:490). The GH component (LabRequests.Lab) passes `cache_dir` only when it is non-empty and the method is bem.
- A run resumes when it uses the same folder and the same key. The key hashes the geometry, the directions, the frequencies, c and the edges (bem.py:540). Finished NumCalc steps stay on disk after a cancel, and manage_numcalc skips them.
- `generation["cache"]` in the result gives the exact folder a run used.

## Tried and ruled out
- "cache_dir is needed to be able to resume": wrong. The default folder resumes just as well.

## Watch out
- Changing `cache_dir` between runs (set to empty, or the reverse) points the second run at the other folder, so it starts over and the first folder is orphaned.
- Nothing cleans the default cache. NumCalc projects are large, so point heavy runs at a drive with space.
- Not verified: whether the GH worker process sees the user's normal home directory for `~`.
