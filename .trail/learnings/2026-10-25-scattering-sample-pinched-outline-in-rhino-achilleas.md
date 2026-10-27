---
title: "Scattering Sample 'pinched outline' in Rhino: a closed tile in disc mode"
author: Achilleas
date: 2026-10-25
branch: feat/coefficient-lab
cost: { minutes: 4, tokens: 82200 }
---

## Decided
- The 'found a pinched outline' error on CA_Scattering_Sample with a closed stepped block (0.2 x 0.2 m, 10 mm plate + 4 blocks, meshed by GH Mesh) came from mode = disc. ScatteringSample.disc tiles the tile and welds it; a closed tile's side walls coincide with the neighbour's, so edges are used by 4 triangles and _boundary_loops raises 'pinched'. The same mesh passes in as_drawn (d 0.28 m), checked through the live worker (Worker.Instance.Call via reflection from lamcp).
- The worker venv (~/.compas_acoustics/worker) runs the same sample.py as the repo here; a local repro with the mesh exported from Rhino behaves identically.

## Tried and ruled out
- Stale solve, C# MeshOf/ToMeshData conversion, the diameter kwarg in as_drawn (ignored by make_scattering_sample): none of these is the cause.
- Making the tile open by deleting its bottom (or bottom + every face on the tile sides) still pinches: the side faces run from z 0 to the block top as single faces, and the blocks touching the tile edges need their seam step wall kept exactly once.

## Watch out
- The component's Message strip is not updated when the worker raises: it kept 'as_drawn, d 1.41 m' from an earlier solve while mode was disc. Read the real mode from the hidden port: component.HiddenValue('mode') (the _choices dict is empty for hidden control ports).
- The worker has no log file; ~/.compas_acoustics/workers/<pid>.json only holds the port.
