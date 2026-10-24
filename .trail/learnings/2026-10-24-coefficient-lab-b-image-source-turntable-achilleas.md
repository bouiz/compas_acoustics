---
title: "Coefficient lab B: image-source turntable is infeasible; ray-traced absorption needs a sample-scaled chamber"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 13, tokens: 669236 }
touches:
  - docs/superpowers/plans/2026-10-23-coefficient-lab-v1-a-scattering-core.md
  - docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md
  - src/compas_acoustics/lab/scattering.py
  - tests/lab/test_scattering.py
---

## Decided
- ISO 354 absorption in the virtual chamber is done by ray tracing (Mono ray_tracing=True); the chamber must be scaled to the sample (volume ~ 200 m3 x (d/3 m)^3, ISO 17497-1 §6.3.1 / ISO 354 S/V^(2/3) range), because T and areas scale with length but alpha does not. In a 200 m3 chamber a 0.8 m disc changes A by ~1.5 %, below the ray tracer's ~4 % seed-to-seed spread of T.
## Tried and ruled out
- Virtual ISO 17497-1 turntable with image sources only (Mono ray_tracing=False): the standard evaluates the decay from -5 to -20 dB, which for a ~3 s chamber is >1 s of response. Measured (7.2x5.3x5.2 m, walls alpha 0.05): the IR grows ~0.02 s per image order, so order ~50 is needed; cost with a closed 12-face obstacle is 4.5 s at order 7 (empty box: 0.0 s at order 6, 1.1 s at order 12), exploding with order; x72 positions x pairs x 2. The EDC of a truncated IR reaches -20 dB near its end, which looks like a decay but is not.
## Watch out
- Ray-traced responses with the sample as an obstacle cost ~50-105 s per run for 2.4k-8k faces at 20000 rays (empty chamber 1.3 s).
- AcousticAnalyser.reverberation_times works in the 7 octave bands, not third octaves; the lab needs its own one-third-octave decay evaluation.
