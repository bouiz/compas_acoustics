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
| `hybrid_reference_82833d1.npz` | a 40 ms `Hybrid` response (32 kHz, sh_order 1) from the code before surface scattering, for `test_without_scattering_nothing_changes`; made by `hybrid_reference.py` run against `git archive 82833d1` | this package at 82833d1 |
| `hybrid_reference_f658127.npz` | the same scene's ambisonic responses from the code before the hybrid rendered ambisonic and binaural outputs from one run: `rir` (Eyring tail), `scattering` (`scattering=True`) and `ray_tracing` (a seeded ray-traced reference, so the late envelope is shaped), for `test_the_ambisonic_output_is_unchanged`; made by `hybrid_reference.py <out> rir scattering ray_tracing` run against `git archive f658127` | this package at f658127 |
| `razr_scattering/` | surface scattering (Ewert et al. 2025): the split per image, the time-spread delays and the coupling into the FDN; made here, with its script, see its README | RAZR 0.96.3, MATLAB R2025b |

Licence: RAZR is CC BY-NC-ND 4.0, and `razr_scattering/` holds only numbers derived from its output. Otherwise not confirmed. Check the terms of EDtoolbox and the UDFA toolbox, and ask their authors, before redistributing these
files beyond this repository.
