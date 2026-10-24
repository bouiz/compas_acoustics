---
title: "Coefficient lab: proxy area tolerance 1e-2 and the backing warning resolve two parked items"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 21, tokens: 123331 }
---

## Decided
- The proxy footprint check (ScatteringSample.as_drawn(..., reference=)) now accepts bottom areas within 1e-2 relative (was 1e-3, user decision 2026-10-24): a circle tessellated with another segment count differs by about 1e-3 (64 vs 128 segments 1.2e-3, 64 vs 48 and 64 vs 32 also accepted). A square over a disc (21 %) and a 16-gon over a 128-gon (2.5 %) are still refused. The bounding-box tolerance stays 1e-3 of the diameter.
- The geometric method warns when a closed mesh stands on a backing thicker than baseplate: GeometricResponse.height_mean and .height_spread; the warning fires when height_spread > 1e-6 m and height_mean > 2 * height_spread. Verified remedy: as_drawn(mesh, baseplate=<backing thickness>) gives the same s per octave as the relief drawn open (0.001, 0.002, 0.01, 0.038, 0.141, 0.444 against 0.001, 0.002, 0.01, 0.038, 0.140, 0.442), while the default baseplate gives 0.054, 0.196, 0.564, 0.942, 0.999, 0.999.
## Tried and ruled out
- Warning on any mean above 0: a box has spread 0 and its height really is its depth; an open sawtooth has mean 0.5 H below 2 sigma = 0.577 H. Only a positive mean (a negative one is an object below its reference, the remedy does not apply).
## Watch out
- Supersedes the two "Parked, not fixed" items of the learning "Coefficient lab D1+D2: geometric method, proxy reference, diffusion coefficient". The other parked items there still stand (_inside cost on fine outlines, rays bouncing forever at shared edges, _clean fanning non-convex n-gons).
- The warning text says "mounting plane z = 0" also for a proxy reference.
- trail check on this branch compares ~500 files against the learnings and flags 13 lines by keyword (image source, first hit, rigid kirchhoff ...) in untouched C# tests, README and old CHANGELOG entries: false positives, none from these changes.
