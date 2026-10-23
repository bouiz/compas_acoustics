---
title: "Coefficient lab: band range, source-preset defaults and Kirchhoff form (user decisions)"
author: Achilleas
date: 2026-10-23
branch: feat/coefficient-lab
cost: { minutes: 27, tokens: 143120 }
touches:
  - docs/superpowers/plans/2026-10-23-coefficient-lab-v1-a-scattering-core.md
  - docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md
  - CHANGELOG.md
  - csharp/CompasAcoustics.Grasshopper.Tests/ArchiveCompatibilityTests.cs
  - csharp/CompasAcoustics.Grasshopper.Tests/PortDefaultsTests.cs
  - csharp/CompasAcoustics.Grasshopper/Components/SimulationComponents.cs
  - csharp/CompasAcoustics.Grasshopper/Components/SimulationSettings.cs
  - csharp/CompasAcoustics.Grasshopper/Components/Specs/CA_Simulate_Mono/metadata.json
  - src/compas_acoustics/rays.py
  - src/compas_acoustics/simulation_methods/hybrid.py
---

## Decided
- Bands: one-third octaves 100-5000 Hz by default (ISO 354 / ISO 17497-1 / -2 all specify this range); optional extension to 8000 Hz flagged beyond ISO. The 8 kHz material octave takes the 4 kHz value in the default range (warned). Cost model: BEM 0.41x (default) vs 1.00x (to 8 kHz) for a 0.8 m disc; 100-10k would be 1.86x and hit NumCalc's ~100k element limit.
- Source preset default depends on the method: kirchhoff -> "fine" (Heimes 864 dirs), bem -> "standard" (97). Method comparisons must pass the same preset.
- Kirchhoff uses the Beckmann obliquity n.(u + r), not the rigid n.r: reciprocal (needed by V3's s_d reciprocity check); s differs by <= 0.02 on Heimes' sine.
## Tried and ruled out
- Preset "survey" (ISO 17497-2 Table 1, 13 dirs) as default: up to 0.29 error in random-incidence s near grating-lobe onsets (its 60 deg ring carries half the Paris weight).
- Rigid n.r Kirchhoff form: ~20 % non-reciprocal.
## Watch out
- ISO 17497-2 requires >= 3 spectral points per one-third-octave band; one point per octave aliases the narrow notches in s.
