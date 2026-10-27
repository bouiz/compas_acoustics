---
title: "BEM lab at 1-2 kHz on a 1.76 m sample: RAM serialises the run, steps unconverged, 4 kHz will not fit"
author: Achilleas
date: 2026-10-26
branch: feat/coefficient-lab
cost: { minutes: 7, tokens: 152553 }
---

## Decided
Diagnosis only (2026-10-26, A24 run, d = 1.76 m, standard preset 97 sources x 9 frequencies per split); the user decides. Sample elements per split: 1854, 3810, 10894, 39580, 150066 (split_5, 1467-2713 Hz; reference 34820). Split times: 3 min, 11 min, 37 min, 5.5 h. In split_5 NumCalc's RAM estimate for the sample is 7.1-13.8 GB per step, so _numcalc.max_instances = 0.9 x 14.5 GB available // (1.05 x 13.8) -> 1: one NumCalc at a time, 1 of 14 cores. Steps take 7-16 min, so the sample alone is about 873 x 12 min ≈ 7 days, then 873 reference steps (~1 GB, ~70 s each) about 17 h serial.
Convergence: split_4's sample had 667/873 steps at the 250-iteration cap (worst residual 1.8); split_5 7 of the first 9. Splits 1-3 and every reference converged. Capped steps are the slowest steps and their s is unreliable.
Split 6 (4 kHz octave, if in the range) would be ~4x split_5's elements (~570k, under the 10 x max_elements stop) and ~50+ GB per step on a 36 GB Mac: manage_numcalc would stop with "not sufficient free RAM" after the week of split_5.
Levers: elements ∝ d² f² (diameter halved -> 1/4 elements and RAM); steps ∝ sources (survey = 13, standard = 97 without symmetry); elements_per_wavelength 6 -> 5 = 1/1.44. Code: a RAM-budget scheduler (sum of the running steps' own estimates instead of one cap from the largest) would run the 1 GB reference steps ~12 at once, about 15 h saved in split_5; it cannot parallelise the 13.8 GB sample steps.

## Tried and ruled out
More launch speed (_WAIT, max_cpu_load): irrelevant once steps take minutes; RAM is the limit from split_4 on (max_instances 4, then 1).

## Watch out
NumCalc is single-threaded, so one instance = one core whatever the machine.
