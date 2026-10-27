---
title: "A24 GHX (LLH roof lower chords): acoustic method reconstructed"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 10, tokens: 74891 }
---

## Decided
- The acoustic code for ~/Desktop/A24_Stochastic_Lower_Chord_Paths.ghx is NOT in the file: it ran through the LAMCP bridge and only results are stored (GH ValueTable key A24_ACOUSTICS_DATA, JSON). The "Reflection and wave estimate" component only bakes that JSON.
- Reconstructed and checked against the stored values, exact to printed precision. Rays: mirror reflection of each lower-face facet normal at +Z incidence, weight = area*|n_z|; peak cone = max weight share within 10 deg of any facet's reflection direction. Wave: coherent Kirchhoff/Fraunhofer facet sum, I(s)=|sum A*sinc(qu*L/2)*sinc(qv*t/2)*(|s.n|+|d.n|)/2*exp(i q.p)|^2, q=k(s-d), c=343, 16384-point Fibonacci lower hemisphere, 500/1000/2000 Hz, normal incidence only.
## Tried and ruled out
- Incoherent sum, and cos or no obliquity factor: none reproduce the stored metrics.
## Watch out
- Far-field plane-wave metrics give identical values for the concave (UP) and convex (DOWN) vaults, so focusing is invisible. 50 mm strips are sub-wavelength, so Kirchhoff is invalid. The flat reference already has 20-32% "outside cone" (finite aperture plus grating lobes).
