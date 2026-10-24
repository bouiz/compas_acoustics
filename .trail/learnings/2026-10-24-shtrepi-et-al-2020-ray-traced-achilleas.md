---
title: "Shtrepi et al. 2020 ray-traced diffuser method: evaluated against the coefficient lab"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 2, tokens: 114527 }
---

## Decided
- Shtrepi, Mendez Echenagucia, Badino & Astolfi (Building Acoustics 2020, doi 10.1177/1351010X20967821) compute a frequency-independent uniformity index u_theta (ISO 17497-2 eq. 6 on ray-traced levels, two orthogonal semicircles at 5 m, point source at 10 m, specular-only rigid panel). They do NOT compute the scattering coefficient s (paper p. 3). At most it is a fast qualitative preview in the lab, never the source of s.
- Proposed (not yet approved by the user): a third scattering method `sbr` (shooting and bouncing rays + physical-optics exit integral, Ling/Chou/Lee 1989), coherent, multi-bounce, cost independent of lambda, feeding the existing Mommertz correlation pipeline; and a normalised diffusion coefficient d_n output from the same pressures.
## Tried and ruled out
- Using the paper's u_theta as a design metric for ranking: their own data puts flat at 0.31 (BEM d ~0.02-0.05), rectangular below flat, curved above sawtooth, contrary to BEM, because there is no flat-reference normalisation and phase is absent.
## Watch out
- Their overlapping spherical receivers (radius = spacing, eq. 2-3) do not conserve energy: a ray through a centre gives total chord 2r, through a midpoint 2*sqrt(3)*r (~73 % overcount), not the "half/half" the text claims. At 1 deg / 5 m / 50k rays only ~8 rays land per receiver, matching the Table 1 spread (curved 0.31 -> 0.45).
- For objects up to 10 m, BEM is infeasible at high frequency and even Kirchhoff/SBR need Delta-theta <= 50*lambda/d (~0.34 deg at d=10 m, 5 kHz, ~1e5 receivers): simulate a representative disc for relief surfaces; freestanding objects belong in the room geometry, not in an area-based s.
- Lab s must go on a flat proxy surface in room simulations, never on the detailed geometry as well (double counting).
