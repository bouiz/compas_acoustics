---
title: "Coefficient lab on an open two-layer timber truss: which method, and what to propose"
author: Achilleas
date: 2026-10-26
branch: feat/coefficient-lab
cost: { minutes: 5, tokens: 77427 }
---

## Decided
- Assessed (no code changed): none of geometric / kirchhoff / bem gives an accurate s for an open lattice truss (top slat layer + V web + bottom chords) as-is.
- Geometry: separate (un-unioned) members fail ScatteringSample.as_drawn: closed path -> _bottom_loop finds one loop per bottom chord ("found N outlines"); overlapping solids fail _to_body or are wrong for Kirchhoff/BEM. Prep: boolean-union all members + a plate over the footprint, room side to +Z, as_drawn(mesh, baseplate=<plate thickness>).
- If the truss is fixed against a rigid slab: BEM, preset "standard", band_range "iso", cache_dir on a big drive; full truss to the 1 kHz octave, a representative crop (>= 2-3 slat and V periods) for 2-4 kHz.
- If it hangs free (open above): lab is out of scope (mounted only). Recommend modelling the truss as room geometry; a free-field, no-baffle, full-sphere BEM mode (reflected/transmitted split + s of the reflected part) was proposed, not approved.

## Tried and ruled out
- geometric: heuristic, one h for a two-layer non-height-field object, no diffraction. kirchhoff: single bounce, no inter-layer reflections or gap diffraction, members ~50 mm are sub-wavelength.

## Watch out
- BEM size: edge c/(6 f_max) ~ 10 mm in the top split -> ~45k remeshed elements per m^2; ~100k elements ~ 10 GB per step; a full truss has ~20-40 m^2 of member surface. 1 kHz octave ~3k/m^2.
- AcousticMaterial.transmission_coefficients exists but nothing in src/ beyond materials.py and pb/conversions.py reads it: a transmission-based proxy needs engine work too.
