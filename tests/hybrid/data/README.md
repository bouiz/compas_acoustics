# Oracle data

Numerical results of the authors' MATLAB toolboxes, for the tests of the diffraction code. The files hold results (levels and
sound pressures for generated geometries), not toolbox code. They are copied unchanged from hybrid-rir at the revision in
`../UPSTREAM_REVISION`. The MATLAB scripts and case generators that made them are not in this repository.

| File | What | Made with |
|---|---|---|
| `udfa_single_oracle.mat` | 180 single-wedge cases (10 per group, seed 20261003), `p_udfa` and `p_btms` | UDFA toolbox v0.90 (Kirsch and Ewert), EDtoolbox v0.504 (Svensson), MATLAB R2025b |
| `double_edge_oracle.mat` | the three-sided barrier (Kirsch and Ewert 2024) against second-order BTMS and the UDFA toolbox | same |
| `btms_first.mat`, `btms_second.mat` | first and second order BTMS cases (240 and 80, seed 20261006) | EDtoolbox v0.504, MATLAB R2025b |
| `btms_plate.mat`, `btms_fd.mat` | a plate, and the second-order frequency-domain reference of three boxes and a prism | EDtoolbox v0.504 |
| `hybrid_reference_2535c3d.npz` | 40 ms ambisonic `Hybrid` responses (32 kHz, sh_order 1) of one scene: `rir` (Eyring tail), `scattering` (`scattering=True`) and `ray_tracing` (a seeded ray-traced reference, so the late envelope is shaped), for `test_the_ambisonic_output_is_unchanged` and `test_without_scattering_nothing_changes`; made by `hybrid_reference.py <out> rir scattering ray_tracing` run against `git archive 2535c3d`, the code that gave the late envelope its directional shares (`rir` and `scattering` equal the cb644a8 file; earlier files: cb644a8 with 96 lines and the hemisphere drive, 82833d1 before surface scattering, f658127 before the binaural output) | this package at 2535c3d |
| `ambisonic_reference_3b9333d.npz`, `binaural_reference_3b9333d.npz` | 40 ms `Hybrid` responses (32 kHz) of the same scene in both formats over ten cases (`hybrid_reference.CASES` and, ambisonic only, `AMBISONIC_ONLY`: plain, scattering, a ray-traced reference, two slopes, no directional absorption, UDFA and BTMS with an obstacle panel, a turned receiver, orders 2 (N3D) and 3), the regression oracles of the split into the binaural and ambisonics methods; made by `hybrid_reference.py <out> --format ambisonic|binaural` on this tree at 3b9333d | this package at 3b9333d |
| `razr_scattering/` | surface scattering (Ewert et al. 2025): the split per image, the time-spread delays and the coupling into the FDN; made here, with its script, see its README | RAZR 0.96.3, MATLAB R2025b |

Licence: RAZR is CC BY-NC-ND 4.0, and `razr_scattering/` holds only numbers derived from its output. Otherwise not confirmed. Check the terms of EDtoolbox and the UDFA toolbox, and ask their authors, before redistributing these
files beyond this repository.
