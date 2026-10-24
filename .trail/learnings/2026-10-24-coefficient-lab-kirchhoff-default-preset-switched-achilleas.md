---
title: "Coefficient lab: Kirchhoff default preset switched to standard"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 30, tokens: 736298 }
touches:
  - docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md
  - requirements.txt
  - src/compas_acoustics/lab/grids.py
  - src/compas_acoustics/lab/solvers/bem.py
  - tests/lab/test_bem.py
  - src/compas_acoustics/lab/results.py
  - src/compas_acoustics/lab/scattering.py
  - tests/lab/test_scattering.py
  - src/compas_acoustics/lab/solvers/kirchhoff.py
  - tests/lab/test_kirchhoff.py
---

## Decided
- measure_scattering's default source preset is "standard" (97 directions) for both kirchhoff and bem (user, 2026-10-24). This supersedes the earlier "kirchhoff -> fine" decision.
## Tried and ruled out
- Kirchhoff default "fine" (864 directions): measured ~45 min on 14 cores for a 0.8 m disc even after parallelising (~19 h single-threaded before), vs ~5 min for "standard"; too slow for the "fast" method.
## Watch out
- Comparisons between methods must still pass the same preset explicitly.
