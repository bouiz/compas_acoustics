---
title: "Coefficient lab C re-checked in Grasshopper: balloon valid, heuristic notes are Remarks"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 18, tokens: 90225 }
touches:
  - tests/test_worker_api.py
  - CHANGELOG.md
  - csharp/CompasAcoustics.Grasshopper.Tests/LabComponentTests.cs
  - csharp/CompasAcoustics.Grasshopper.Tests/WorkerIntegrationTests.cs
  - csharp/CompasAcoustics.Grasshopper/Components/ComponentBase.cs
  - csharp/CompasAcoustics.Grasshopper/Components/LabComponents.cs
  - csharp/CompasAcoustics.Grasshopper/Components/LabResults.cs
  - src/compas_acoustics/lab/scattering.py
  - src/compas_acoustics/worker/api.py
  - csharp/CompasAcoustics.Grasshopper/Components/Specs/CA_Polar_Preview/metadata.json
---

## Decided
- Supersedes the DEFECT in "Coefficient lab C checked in Grasshopper through lamcp": polar_balloon now keeps a 3 % minimum radius (radius * (0.03 + 0.97 * clip(1 - level/floor_db, 0, 1))). Re-checked live in Rhino on the sparse geometric result (205 of 207 cells empty): the mesh is valid (1197 vertices, 399 faces, none at the origin, no zero-area faces, radii 0.03..0.958) and draws as the two reflection lobes.
- The geometric heuristic notes (Lab's `_HEURISTIC`, Lab Material's) are returned by the worker under `notes` (exact-text split, `HEURISTIC_NOTES` in lab/scattering.py and `MATERIAL_HEURISTIC_NOTE` in worker/api.py) and shown as Remarks; `result.warnings` and the Python API still carry them. Lab Material has no warning on a geometric run; the Lab keeps one real Warning, "the 8000 Hz octave was not simulated", on every default-range run (kept as a Warning on purpose).
## Tried and ruled out
- Pinning the balloon only at its endpoints: three formula mutations (squared fraction, upper clip 2.0, floor_db*1.1) passed every test; the dense test now covers +3..-60 dB at two floors and fails each mutation. The wire rounds polar levels to 0.1 dB, so the interior check cannot see an error below ~0.2 % of the radius.
## Watch out
- Two implementers must not edit tests/test_worker_api.py or worker/api.py at once; queue the second.
- The lamcp script globals hold `objs` for the bridge: use other names; the bridge needs its toggle switched on again after every Rhino restart.
