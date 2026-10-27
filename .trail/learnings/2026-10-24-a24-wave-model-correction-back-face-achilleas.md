---
title: "A24 wave model correction: back-face culling, and common-dB fan outputs"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 11, tokens: 87562 }
---

## Decided
- Correction to the earlier A24 reconstruction: the stored wave model also drops facets facing away from each outgoing direction (term (s.n>0) with n pointing down). With this term all 54 trial and baseline outside-cone values match to 2e-10. Without it, strong directions still match but weak and grazing ones are off by up to ~11 dB.
- Baseline (REFERENCE) facets can be rebuilt exactly from A24_MATRIX_DATA baseline rows. Facet = bottom edge of the LOWER-row side_polys[0]: p is the edge midpoint at y_start+t/2, u is along the edge, n=(u_z,0,-u_x). Baselines are identical across the 6 variants.
- Common-scale fans: directivity 10log10(2*pi*I/P), P = power summed over the hemisphere, so 0 dB means uniform spread. Plot range -15..+35 dB (peaks reach +34 dB at 2 kHz on the flat reference). Scripts and outputs are in ~/Desktop/A24_common_scale_fans/.
## Watch out
- The stored 'plot' entries in A24_ACOUSTICS_DATA are scaled to each cell's own maximum. They cannot be put on a common scale, so recompute from the facets.
