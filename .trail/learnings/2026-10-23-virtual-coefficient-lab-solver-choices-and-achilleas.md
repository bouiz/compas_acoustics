---
title: "Virtual coefficient lab: solver choices and what the Heimes scripts do"
author: Achilleas
date: 2026-10-23
branch: feat/coefficient-lab
cost: { minutes: 54, tokens: 439400 }
touches:
  - docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md
---

## Decided
- Scattering coefficients for 3D surfaces need phase-coherent pressures: methods `bem` (mesh2scattering/NumCalc, reference), `kirchhoff` (fast physical optics, same correlation pipeline), `turntable` (virtual ISO 17497-1, image sources only). Absorption: ray-traced virtual ISO 354 chamber. Spec: docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md
- GH side is C# components + compas_pb messages (new messages in acoustics.proto), never CPython components.
- Heimes & Vorländer (Zenodo 15673498): s_d = 1 - mesh2scattering.process.scattering_freefield(sample, reference rotated by azimuth+180°, receiver weights) per output direction; random-incidence s = 1/π ∬ s(θ,φ) cos²θ sinφ; sources plane waves at colatitude 1.26°..88.74° step 2.52°, receivers pyfar sph_equal_area + Voronoi weights, z>0.
## Tried and ruled out
- Ray-traced ISO 17497-1 with the pyroomacoustics ray tracer: its tail is a random-sign Poisson sequence, identical across runs with seed=True, so phase-locked averaging is meaningless. Use image sources only (ray_tracing=False).
## Watch out
- mesh2scattering needs Python >= 3.11; the worker process can use it, Rhino's CPython 3.9 cannot.
- AcousticMaterial only accepts the 7 octave bands 125 Hz-8 kHz; fold 1/3-octave results by arithmetic mean of 3.
