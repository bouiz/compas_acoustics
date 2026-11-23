---
title: "Lab CI on PR #38: tests that need trimesh, and the flaky Kirchhoff cancel test on Linux"
author: Achilleas
date: 2026-11-23
branch: feat/coefficient-lab
cost: { minutes: 15, tokens: 79224 }
touches:
  - src/compas_acoustics/lab/solvers/kirchhoff.py
  - tests/lab/test_kirchhoff.py
  - ".github/workflows/build.yml"
  - tests/test_pb.py
  - tests/test_worker_api.py
  - docs/_images/components/CA_Polar_Preview.png
  - docs/_images/components/CA_Scattering_Lab.png
  - docs/_images/components/CA_Scattering_Sample.png
  - docs/_images/fingerprints.json
  - requirements-lab.txt
---

## Decided
- 2026-11-24, PR #38 (feat/coefficient-lab): the CI build installs `dev,lab` (compas-actions/ci input `extras`). requirements-lab.txt limits every lab package to Python >= 3.11, so on 3.9/3.10 the extra adds nothing. Tests that run the geometric method outside tests/lab carry `needs_raycaster` (test_worker_api.py) or `pytest.importorskip("embreex")`.
- KirchhoffSolver's `_kill_pool` (cancel): list `executor._processes` BEFORE `shutdown()`, read the table again after, terminate, then join and kill each process, and finally join `executor._executor_manager_thread`, grabbed before shutdown, which shutdown sets to None. With 6c6e3da all 17 checks passed, and Ubuntu 3.9 and 3.10 passed 3 runs each.

## Tried and ruled out
- Fixes that failed on CI one after another: terminate + kill only (b290432); reading the table only after shutdown (213c1da: the manager thread pops a worker that is exiting and joins it there, see CPython concurrent/futures/process.py process_result_item); before + after without joining the manager thread (2047b40). The last race: the manager's blocking os.waitpid reaps a worker, our poll gets ECHILD, and multiprocessing reads the worker as alive until the manager stores its exit code.
- Counting every `multiprocessing.active_children()` in the test: pytest-xdist runs other tests in the same process. Count only the children started since the test began.

## Watch out
- The cancel failure is intermittent and Linux-only, and doesn't reproduce on macOS. Judge a fix by several CI runs: re-run one job at a time, since `gh run rerun --job` refuses a second job while the first is queued.
- The test's assertion names each leftover's (pid, exitcode). An exit code that isn't None means a reaped worker, not a live one.
