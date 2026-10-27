---
title: "BEM results could not be written: netCDF4 >= 1.7.4 breaks sofar's string variables"
author: Achilleas
date: 2026-10-26
branch: feat/coefficient-lab
cost: { minutes: 5, tokens: 43558 }
touches:
  - requirements.txt
  - src/compas_acoustics/lab/solvers/bem.py
---

## Decided
2026-10-26, the first BEM run to finish a split failed in BemSolver._read -> mesh2scattering.output.write_pressure -> sofar.write_sofa -> netCDF4.stringtochar(value, encoding="utf-8"): AttributeError: 'numpy.bytes_' object has no attribute 'encode'. The string variable is mesh2scattering's SampleSymmetryAzimuth. Bisected in scratch venvs: netCDF4 1.7.3 works (with numpy 2.3 or 2.4.6); 1.7.4, 1.7.5 and 1.7.5.1.post1 (the latest) fail with either numpy. Fix: `netCDF4 <1.7.4` in requirements.txt (sofar is a base requirement, so the pin is too). The worker had 1.7.5.1.post1 and the dev env 1.7.3, which is why the tests never saw it.
Second fix: _read removes the .sofa when write_pressure raises. The failed write had left a 78 MB partial file, and every later run read it (ValueError in chartostring) instead of writing it again.
Tests: test_sofa_with_string_variables_round_trips (needs a string variable; GLOBAL_* are attributes and do not reach stringtochar), test_a_failed_pressure_write_leaves_no_partial_file.

## Tried and ruled out
numpy 2.4 as the cause: netCDF4 1.7.3 with numpy 2.4.6 round-trips.

## Watch out
The worker service imports sofar/netCDF4 in-process, so after changing them it must be restarted (Worker component's Restart button); _numcalc changes need no restart, bem.py changes do.
The installer installs from dist/compas_acoustics_installer/*.whl: rebuild it, or a fresh install pulls netCDF4 1.7.5 again.
The worker venv has no pip: use `uv pip install -p ~/.compas_acoustics/worker/bin/python ...`.
