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

Licence: not confirmed. Check the terms of EDtoolbox and the UDFA toolbox, and ask their authors, before redistributing these
files beyond this repository.
