---
title: "Coefficient lab D1+D2: geometric method, proxy reference, diffusion coefficient (decisions and pitfalls)"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 28, tokens: 80795 }
touches:
  - docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md
  - src/compas_acoustics/lab/geometric.py
  - src/compas_acoustics/lab/raycast.py
  - src/compas_acoustics/lab/scattering.py
  - tests/lab/test_geometric.py
  - tests/lab/test_raycast.py
  - tests/lab/test_scattering.py
  - src/compas_acoustics/lab/sample.py
  - tests/lab/test_sample.py
  - docs/superpowers/plans/2026-10-14-audio-player-csharp.md
---

## Decided
- measure_scattering(method="geometric") is now the DEFAULT (user decision 2026-10-24); kirchhoff, bem (and a planned sbr) are options. Needs trimesh + embreex, Python >= 3.11; no silent fallback, the ImportError names method="kirchhoff". Objects are MOUNTED (user): 1 to 10 m, deep cavities allowed; free-standing objects out of scope.
- Engine (lab/geometric.py): specular rays aimed at the reference's flat BOTTOM face footprint at z = 0, rigid baffle at z = 0 OUTSIDE the footprint only (inside, rays reflect off whatever they hit, also below z = 0), exits incl. d_z == 0 binned on ReceiverGrid.hemisphere(bin_deg=10). s_max = 1 - sum(min(E_sample, E_reference)) with the LOST fraction appended as one extra bin. Per-octave s = s_max * (1 - exp(-(k h)^2)), h = max(2 sigma_z, |mean_z|) of the height difference sample - reference (2 sigma alone gave h = 0, hence s = 0, for any flat-topped box).
- Proxy reference: as_drawn(mesh, reference=proxy); must share bounding box AND bottom area (1e-3) with the sample; every error names "the proxy"; results flagged non-ISO. d (ISO 17497-2 eq. 6, weighted) returned for every method.
- Accuracy record (geometric vs kirchhoff, Heimes sine, survey preset): max |delta| per octave 0.001 / 0.003 / 0.016 / 0.063 / 0.191 / 0.186 (125...4000 Hz); heuristic over-estimates from 500 Hz up; 28x faster. The 2 sigma depth factor is an open question for the owner.
## Tried and ruled out
- Lost rays counted as scattering when lost in BOTH bodies (rim walls): append lost as an extra bin. Baffle rule "first hit below z = -1e-9": wrong for closed meshes (lowest point at -baseplate); use the footprint test. Rays leaving exactly horizontally as "lost": bin them (they graze the baffle).
- RayCaster without `exclude`: grazing rays (elevation < 1e-2) self-hit the face they left; always pass the previous hit triangle. trimesh has no ray backend here without embreex or rtree.
## Watch out
- Parked, not fixed: (1) a closed mesh with a backing thicker than `baseplate` gets the backing height as depth (sawtooth on a 0.1 m plate: s at 1 kHz 0.94 vs 0.038 open); user remedy as_drawn(mesh, baseplate=<backing>); a warning when sigma > 0 and |mean| > 2 sigma is still to do. (2) Proxy area tolerance 1e-3 refuses a circle tessellated with another segment count (64 vs 128 segments is 1.2e-3); recommend 1e-2. (3) _inside costs ~47 s on a 1000-segment outline at the standard preset (no bbox pruning). (4) ~0.04 % of rays bounce forever at shared edges. (5) _clean fans n-gons from the first vertex: non-convex closed n-gon bottoms can be wrong.
- The Heimes-sine Kirchhoff cross-check hung once in a worker pool (multiprocessing worker went defunct); run it with workers=1 if it does.
