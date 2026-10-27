---
title: "BEM lab launch wait 0.25 s measured: 7.5 instances, launch loop still the limit"
author: Achilleas
date: 2026-10-26
branch: feat/coefficient-lab
cost: { minutes: 6, tokens: 5359 }
---

## Decided
Measured 2026-10-26 on the A24 run, split_1 (3 s steps), after _WAIT went to 0.25 s: 146 launches in 67 s (2.2/s, was 0.47/s), 6-10 NumCalc instances (mean 7.5, was ~1.5), system CPU ~77 %. Split_1 goes from ~1 h to ~13 min. That is fewer than the 12-13 instances the earlier learning expected.
Why: each loop of manage_numcalc costs _WAIT + 0.1 s (psutil.cpu_percent(.1)) + a process scan, about 0.45 s, so concurrency ≈ 2.2/s x 3.4 s. Its 90 % CPU gate also stopped launches 10 times (the log showed 9 instances at 91-94 %; the gate measures the whole system over 0.1 s, Rhino included).

## Tried and ruled out
Nothing further changed yet; the next tuning waits on the user: _WAIT ~0.1 s and manage_numcalc(max_cpu_load=~95-98) for ~11-13 instances on the short splits, at the cost of Rhino's responsiveness during the run.

## Watch out
The gain applies only where steps are short (the low splits); long high-frequency steps were never launch-limited.
