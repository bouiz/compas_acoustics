---
title: "BEM lab low CPU: manage_numcalc's 2 s wait between launches caps parallelism on short steps"
author: Achilleas
date: 2026-10-26
branch: feat/coefficient-lab
cost: { minutes: 5, tokens: 64700 }
touches:
  - src/compas_acoustics/lab/solvers/bem.py
---

## Decided
Diagnosis only (2026-10-26); no fix yet, the user decides. NumCalc instances are single-threaded and manage_numcalc launches them strictly one at a time, sleeping wait_time after each launch (_numcalc._WAIT = 2 s, plus ~0.1 s for psutil.cpu_percent). In the low splits a step lasts only ~3 s (A24 run, split_1: 1854 sample elements, 196 reference, 97 sources x 9 frequencies = 1746 steps), so only ~1.5 instances ever overlap: 237 launches in 504 s (2.13 s each), 1-2 NumCalc at 100 % on a 14-core Mac, ~88 MB each, about 9 GB RAM free. max_instances (14) and the 90 % CPU gate were never what held it back. The launch interval is the floor: ~1746 x 2.1 s ≈ 1 h per split, however small the mesh.
Proposed fix: _WAIT around 0.2-0.5 s (safe because max_instances is already bounded by NumCalc's own RAM estimate in _numcalc.max_instances); a steady state of up to ~13 instances then makes the short splits ~8-10x faster. The high splits (long steps, GB of RAM each) gain little: there RAM is the limit.

## Tried and ruled out
RAM or the CPU gate as the limit: the log shows "waiting for resources" only twice in 237 launches.

## Watch out
With a wait near 0, _numcalc_instances() counts processes named NumCalc; the shell=True wrapper (sh) might not yet have exec'd NumCalc, so there can be a brief overshoot of max_instances. Keep a small nonzero wait.
Splits run one after another (BemSolver._run per split), so each split's tail runs at low parallelism; running all splits under one scheduler would remove that, but it matters less.
The Lab runs the worker's installed copy (~/.compas_acoustics/worker/...): reinstall and restart the worker service to test, which kills a running job (finished steps stay in the cache and resume).
