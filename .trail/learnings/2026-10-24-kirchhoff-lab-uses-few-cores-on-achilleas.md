---
title: "Kirchhoff lab uses few cores on large samples: the 512 MB source batch caps the pool"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 4, tokens: 60241 }
---

## Decided
Nothing changed yet (2026-10-25); the fix waits on the user. Diagnosis: KirchhoffSolver already runs sources on a spawn ProcessPoolExecutor (one task per source), but measure_scattering's _pressures feeds it batches of _BATCH_BYTES (512 MB) // per-source pressures, and the next batch is submitted only after the current one finishes. Receivers grow ~d^2 (spacing min(5 deg, 50*lambda/d)), so a 4.7 m sample gets batches of 2-6 sources and only that many pool processes ever start (seen in ps: 2 spawn_main at 100 % on a 14-core Mac, the service at 0 %). Batch sizes: d=0.8 iso 236, d=2.0 iso 37, d=3.0 iso 16 / extended 6.
Proposed fix: one pool task per (source, receiver block) so that every worker stays busy even with a batch of one source; memory stays under the cap.

## Tried and ruled out
Raising the batch to at least `workers` sources: about 14 x 250 MB plus as much again in temporaries for large samples.
Threads instead of processes: GIL-bound Python/NumPy mix; processes stay.

## Watch out
The Lab runs the worker's installed copy (~/.compas_acoustics/worker/lib/python3.12/site-packages), not src/: reinstall and restart the worker service to test, which kills a running job.
estimate_seconds assumes every worker is busy, so until the fix it underestimates large-sample runs.
