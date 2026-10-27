---
title: "Lab user guide: where the facts live and the agreed audience"
author: Achilleas
date: 2026-10-25
branch: feat/coefficient-lab
cost: { minutes: 2, tokens: 49293 }
touches:
  - docs/coefficient-lab-build-notes.md
---

## Decided
- The Lab user guide is for plugin users (designers/acousticians), published as one shared HTML artifact, not Sphinx docs (user's choice, 2026-10-25). Design proposed in chat; awaiting approval before building.
- Sources of truth for the guide: component specs in csharp/CompasAcoustics.Grasshopper/Components/Specs/CA_{Scattering_Sample,Scattering_Lab,Polar_Preview,Lab_Material}/metadata.json (inputs, defaults, dropdown strings); method facts in the docstrings of lab/geometric.py (accuracy table vs Kirchhoff, validity limits), lab/scattering.py measure_scattering (presets, run-time warnings, BEM convergence), lab/solvers/kirchhoff.py.
## Tried and ruled out
- Writing the build pitfalls of docs/coefficient-lab-build-notes.md into the user guide: they are for developers, not users.
## Watch out
- Quote numbers from the code, not memory: geometric over-estimates s by ~0.19 at 2-4 kHz vs Kirchhoff; Kirchhoff fine ~50 min vs standard ~5 min on a 0.8 m disc; BEM is untested in Grasshopper.
- 'scattering' has 7 bands (125-8000 Hz, filled), 'diffusion' pairs with 'octaves' (6 bands with iso), and can be negative.
