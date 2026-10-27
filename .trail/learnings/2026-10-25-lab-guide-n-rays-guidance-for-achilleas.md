---
title: "Lab guide: n_rays guidance for the geometric method, and how to update the guide artifact"
author: Achilleas
date: 2026-10-25
branch: feat/coefficient-lab
cost: { minutes: 9, tokens: 65559 }
---

## Decided
- Advice on n_rays for the geometric method (now in the "How many rays (geometric)" section of the Virtual Scattering Lab Guide artifact, https://claude.ai/artifact/5KU52ACUF6tVxyT4NwbvdJ, v3): default 50 000 per source; 5 000 for iteration; 100k-200k only for large footprints (several m^2) with fine relief or narrow deep cavities. Rule of thumb: ray spacing ~ sqrt(footprint area / n_rays), aim for >= ~10 rays across the smallest feature. Self-check: run at two counts, if s moves < ~0.01 the lower is enough.
- Basis: the accuracy record in the lab/geometric.py docstring (5k vs 50k changed s by <= 0.007 on the Heimes sine; the 0.19 gap to Kirchhoff is the heuristic depth curve, not ray noise). The spacing table is arithmetic, not a measured convergence study.
## Tried and ruled out
- Publishing with `edits` on the guide: refused because a file-edit hook is configured in this setup; Edit the local file, then publish without edits.
## Watch out
- The guide's source is not in the repo. Earlier sessions kept it as scratchpad/lab-guide.html under /private/tmp/claude-503/...; check that a copy matches the live page (Artifact read) before editing, or rebuild from the read.
- The 10-rays-per-feature rule is untested on a real large/cavity sample; a convergence run on one would firm it up.
