---
title: "Coefficient lab C checked in Grasshopper through lamcp: what works, and the invalid balloon of a sparse geometric result"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 8, tokens: 101330 }
---

## Decided
- Task 10 was run live in Rhino 8 (macOS) through lamcp with the installed plugin: the four components register under the planned GUIDs; the chain mesh -> Scattering Sample -> Scattering Lab -> Polar Preview / Lab Material -> Acoustic Object solves without errors; geometric runs on change, kirchhoff only on a Run press (status "running 0:05, kirchhoff 1/702", a 0.2 m disc at preset survey takes ~10 s), a second press re-runs, Cancel shows "cancelled" and stays cancelled; changed inputs without Run give the stale warning and keep the old outputs; a wall-rotated tile with the `plane` input gives scattering identical to the floor tile (max difference 0.0) and without it the readable "footprint has no area" error; dropdown values survive a GH_Archive round trip.
- Drive dropdowns programmatically through the hidden port: `comp.ControlPort(name).PersistentData` (SetChoice only works for `Choices`, which these components do not use); emulate a button with the internal `MarkPressed(port)` via reflection plus `ExpireSolution(True)`.
## Tried and ruled out
- Reading `run_scattering_lab` calls from the worker log: the HTTP handler does not log requests; use the component's own `Runs.AnyRunning`, its Message and its outputs.
## Watch out
- DEFECT: Polar Preview draws nothing for a geometric (default-method) result. At normal incidence 205 of 207 receiver cells are empty (-300 dB); `polar_balloon` maps a level at or below `floor_db` to radius 0, so 1185 of 1197 vertices sit at the origin, every face has zero area and Rhino marks the mesh invalid. A dense Kirchhoff result draws fine. Fix: give the balloon a minimum radius fraction so vertices never coincide, test with the sparse case.
- Not exercised: BEM; Brep/Box/Extrusion/SubD inputs (only a Mesh param was used); a real click on the Run button (emulated); Windows. Minor: the status strip is not cleared on warning paths (Sample keeps "as_drawn, d 1.41 m", Lab keeps its last summary).
- The `objs` name in the lamcp script globals collides with the bridge's own; use other names. A placed chain must start below y = 520 (bridge panel above).
