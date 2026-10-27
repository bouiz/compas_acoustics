---
title: BEM lab launch wait lowered to 0.25 s (follow-up to the low-CPU diagnosis)
author: Achilleas
date: 2026-10-26
branch: feat/coefficient-lab
cost: { minutes: 2, tokens: 13293 }
touches:
  - src/compas_acoustics/lab/solvers/_numcalc.py
  - tests/lab/test_bem.py
---

## Decided
2026-10-26: _numcalc._WAIT went from 2 s to 0.25 s, the fix proposed in the "BEM lab low CPU" learning. Test: tests/lab/test_bem.py::test_numcalc_runner_waits_well_under_a_short_step_between_launches (0 < wait_time <= 0.5). Applied to the worker by copying _numcalc.py into ~/.compas_acoustics/worker/.../site-packages; no service restart is needed, because BemSolver._run starts `python -m compas_acoustics.lab.solvers._numcalc` fresh for every split.

## Tried and ruled out
A wait of 0: manage_numcalc counts instances by process name, so one launched through its shell may not be counted yet, and it could launch more than max_instances.

## Watch out
Not yet measured on a full run; check the concurrent NumCalc count in the next run's split_1 (expect ~12-13 on 14 cores).
