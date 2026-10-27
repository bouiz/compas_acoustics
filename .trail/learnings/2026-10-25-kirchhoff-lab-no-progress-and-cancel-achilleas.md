---
title: "Kirchhoff lab: no progress and cancel ignored while the spawn pool starts its processes one by one"
author: Achilleas
date: 2026-10-25
branch: feat/coefficient-lab
cost: { minutes: 11, tokens: 78971 }
---

## Decided
Diagnosis only (2026-10-25); no fix yet, the user decides. In KirchhoffSolver._run_parallel the line `futures = {executor.submit(...) for i in indices}` runs before the first report() of the wait loop. Python 3.12's spawn ProcessPoolExecutor creates its processes lazily, one per submit(), and each p.start() blocks until the child has re-imported the parent's main module (`-m compas_acoustics.worker.service`, ~6.4 s on an idle machine, mostly pyroomacoustics and pyfar through compas_acoustics/__init__) and read the initargs. So the job thread is stuck for workers x (spawn time) with progress at 0, and jobs.progress() never runs, so a cancel takes effect only after the last spawn. Measured: 7-10 s per child on the 0.8 m sine disc; 30-75 s per child in the user's Grasshopper run while the load average was ~100 (audio daemons, Time Machine). That makes about 8 min of "no progress" and an uncancellable ramp-up. After the ramp, Cancelled -> finally -> _kill_pool did stop all the children.

## Tried and ruled out
BLAS threading in the children: the hot loop (_accumulate) is element-wise NumPy and each child runs about 5 threads.
The installed worker copy being stale: it is identical to src/ for kirchhoff.py, scattering.py, worker/jobs.py and worker/service.py.

## Watch out
Candidate fixes: call report() between submits (cancel and progress during the ramp); a "forkserver" context, so main is imported once and later children fork from the server; lazy imports in compas_acoustics/__init__.
Killing the parent (SIGTERM) leaves the pool children running as orphans, because _kill_pool runs only in the generator's finally.
