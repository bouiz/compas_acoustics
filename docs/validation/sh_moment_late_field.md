# The ambisonic late field from SH energy moments

Validation record of branch `feat/sh-moment-late-field`, which replaces the 96-line FDN of `Ambisonics` with
Gaussian noise shaped by the spherical harmonic energy moments of the rays (`compas_acoustics.hybrid.moments`).
Recorded on top of `d1da8db` (the `engine-integration` tip); the three-room benchmark rerun on upstream v0.2.0
(`dbdf390`). Not a docs page (excluded in `mkdocs.yml`).

## The method

An order-N ambisonic tail has (N+1)^2 channels. Per octave band and 4 ms bin it is described by its channel
covariance

    C_ij = sum_r e_r Y_i(d_r) Y_j(d_r),

over the rays detected in the bin, `e_r` their energy, `d_r` their direction, `Y` the real spherical harmonics (N3D
with W = 1, ACN). A product of two harmonics of order at most N is a sum of harmonics of order at most 2N,
`Y_i Y_j = sum_k G_ijk Y_k` with the real Gaunt coefficients, so

    C_ij = sum_k G_ijk M_k,   M_k = sum_r e_r Y_k(d_r),

and the (2N+1)^2 moments `M` hold everything the covariance holds. `C` is positive semi-definite by construction, a sum
of `e y y^T`. The late field is one white Gaussian noise per channel, split into the octave bands of the
reconstructing bank (which sum back to it), each band mixed per bin by the Cholesky factor of its covariance
(interpolated linearly between bins) and scaled to its energy per sample. The noise is seeded per pair
(`room.raytracing.stable_seed` of the method's seed and the pair's positions).

This reproduces every channel's energy decay and the correlation between channels, which is all of the late field's
spatial information that an order-N signal can carry: its diffuseness, its anisotropy, the direction of its energy. A
dense late tail is close to Gaussian, so these second-order statistics describe it. The early reflections stay exact
image sources (and diffracted paths); only the stochastic tail changes.

### Where the energy and the shape come from

| Late envelope, reference | Energy per band | Directional shape |
| --- | --- | --- |
| `"omni"` (default), reference traced by the method (`tail="ray_tracing"`, or a `Mono` at the method's order) | the reference run's own omnidirectional bins (`room.raytracing.omni_histograms`): exactly what the image sources do not carry | the boundary's reflection `1 - a(u)` per direction (`directional_absorption`), as the FDN's line filters; isotropic without |
| `"directional"`, same reference | the reference run's bins | the rays' moments, the run's rays sorted on the 480-direction moment grid |
| reference given as responses (a dict, a name), or `late_envelope=False`, or an Eyring/Sabine tail | the energy the FDN would radiate in expectation (`network_energy`: the last order's images, and with scattering the scattered parts, decaying at the target RT, one free path later), brought to the reference by the late envelope's gains, which may now raise it up to 12 dB | as the rows above; `"directional"` traces its own histogram on the moment grid |

With the rays' energy (the first two rows) the images keep only their specular part, `prod sqrt(1 - s)`, as with
`scattering=True` and as pyroomacoustics' own images do: the rays carry every reflection's scattered part, from the
first order on, and images keeping the whole reflection counted the early orders' scattered energy twice (+0.22 dB
per order at `s = 0.05`; the hybrid's images were 0.2 to 1.2 dB above the reference's between 0 and 180 ms in the
room with an absorbing floor and ceiling, 0.0 dB after the change, and 0.0 dB in the same room with no scattering).

`late_field="fdn"` keeps the FDN. `generation["late_field"]` records the choice, `generation["late_energy"]`
(`"rays"` or `"network"`) where the energy came from. `Binaural` keeps the FDN.

### Per hit or per grid direction

The pasted design accumulates the moments per ray hit. pyroomacoustics' tracer (libroom, C++, pinned
`>=0.10.1,<0.11`) sorts every detected ray to the nearest direction of a receiver's grid inside its own code and
exposes no hit to Python, so the moments are taken from a fine grid's bins instead: `M_k = sum_g H_g Y_k(u_g)` over
`moment_grid()`, 480 equal-area directions. Against exact per-hit moments of 100 000 hits of an anisotropic field
(`tests/hybrid/test_moments.py`):

| grid | order 1 | order 2 | order 3 |
| --- | --- | --- | --- |
| 96 | 0.85 % | 1.49 % | 1.73 % |
| 240 | 0.31 % | 0.63 % | 0.71 % |
| **480** | **0.18 %** | **0.29 %** | **0.40 %** |
| 960 | 0.09 % | 0.13 % | 0.22 % |

(relative Frobenius error of the covariance). The spread between ray seeds of one 4 ms bin at 100 hits is 22 %
(first order) to 44 % (third), so the grid costs nothing measurable. A per-hit accumulation would need a hook in
libroom (`Microphone::log_histogram`).

### Checks of the synthesis

- The Gaunt reconstruction equals the direct `sum e y y^T` to 1e-15 (orders 0 to 3).
- 2 s of synthesized noise with one anisotropic covariance in every band: W energy within 1.2 % of the target,
  covariance within 1.4 to 2.5 % (the sampling error of 2 s), a flat spectrum.
- The rays' energy is on the image sources' scale: the energy per sample of a white signal at the band's level, of
  which the band takes its share of the spectrum, exactly as pyroomacoustics builds the reference's tail from the same
  bins (`seq_bp * sqrt(hist)`). Normalising each band to unit variance had made the late field 11 to 15 dB too loud.

### After the rays

A band's rays thin out towards the end of the trace: pyroomacoustics drops a ray once it has lost 70 dB
(`energy_thres=1e-7`), and the last bins hold a few rays each. Once a band's rays have decayed 60 dB (`HOLD_DB`,
their own Schroeder integral) or the histogram ends, the late energy continues at the target's slowest reverberation
time and the directional shape is held at its last value. The continuation starts from the trend of the last 150 ms
of rays (`HOLD_FIT_MS`): their own decay per bin from the energy of the window's two halves, their level at the hold
from all of it, faded in over the window. Starting from the last bin instead carried that bin's noise into the
whole continuation: in the corridor the late energy stepped up 5 to 9 dB at the hold and decayed from there.

Where that happens in the corridor (receiver 0, `examples/results/sh_moments/plots/corridor_tail_decay*.png`): at
596 ms (672 ms at receiver 1), 36 to 49 dB below the loudest 50 ms in the 125 Hz to 4 kHz bands, with 0.02 to 0.08 %
of the late energy after it. With the fitted start the moments' decay past the hold lies within 1 to 3 dB of the
FDN's. The reference cannot judge that part: it is built from the same rays and ends where they do.

## Benchmark: three rooms

`examples/benchmarks/sh_moment_late_field.py`, results in `examples/results/sh_moments/summary.md`. Three shoeboxes
of increasing anisotropy, two receivers each, first and third order, all on the same rays (seed 42). The corridor's
duration is set to 2.5 s and the absorbing floor and ceiling room's to 2 s: Eyring predicts 0.5 s in both, and a
reference cut at 0.5 s (28 to 35 dB down in the second room) ends the decay its T30 is measured on.

Means over the receivers. dT30, dEDT (500 Hz - 2 kHz), dC50: the W channel minus the reference. Late level: the late
field's energy minus the reference's late part, mean |dB| 125 Hz - 8 kHz. Cov err: |R - R_rays| / |R_rays|,
`R = C / C_00` over a late window, mean 250 Hz - 4 kHz. Dipoles x/y/z: each dipole's energy relative to W's, 1 in a
diffuse field, mean 250 Hz - 4 kHz (the rays' in brackets). Time: preparing and rendering one pair's response.

| room | N | late field | time s | dT30 s | dEDT s | dC50 dB | late level dB | cov err | dipoles x/y/z (rays) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diffuse office | 1 | moments, omni | 0.51 | -0.007 | -0.025 | +0.40 | 0.90 | 0.32 | 0.99/1.07/1.03 (1.06/1.01/0.93) |
| | | FDN, omni | 2.99 | +0.020 | +0.012 | +1.39 | 2.31 | 0.46 | 1.23/1.25/1.43 |
| | 3 | moments, omni | 0.76 | +0.007 | -0.023 | +0.55 | 1.28 | 0.57 | 1.01/1.10/1.29 |
| | | FDN, omni | 2.96 | +0.020 | +0.012 | +1.39 | 2.31 | 0.75 | 1.23/1.25/1.43 |
| absorbing floor and ceiling | 1 | moments, directional | 1.34 | -0.022 | +0.115 | +0.16 | 0.72 | 0.16 | 1.35/1.51/0.09 (1.43/1.49/0.08) |
| | | FDN, directional | 16.56 | -0.020 | +0.087 | -0.45 | 0.73 | 0.18 | 1.43/1.39/0.26 |
| | 3 | moments, directional | 1.94 | +0.015 | +0.110 | +0.37 | 0.93 | 0.26 | 1.37/1.68/0.09 |
| | | FDN, directional | 16.89 | -0.020 | +0.087 | -0.45 | 0.73 | 0.34 | 1.43/1.39/0.26 |
| corridor | 1 | moments, directional | 1.46 | -0.002 | -0.213 | +0.32 | 2.04 | 0.23 | 3.05/0.05/0.06 (2.89/0.04/0.07) |
| | | FDN, directional | 20.22 | -0.103 | +0.092 | -2.08 | 2.31 | 0.28 | 2.60/0.20/0.24 |
| | 3 | moments, directional | 2.01 | +0.030 | -0.230 | -0.03 | 2.23 | 0.28 | 3.26/0.06/0.08 |
| | | FDN, directional | 20.19 | -0.103 | +0.092 | -2.08 | 2.31 | 0.48 | 2.60/0.20/0.24 |

Over seeds 1 to 4 (first order, directional, 8 responses each, mean +- standard error):

| room | late field | dT30 s | dEDT s | dC50 dB |
| --- | --- | --- | --- | --- |
| diffuse office | moments | +0.009 +- 0.006 | -0.028 +- 0.012 | +0.65 +- 0.31 |
| | FDN | -0.000 +- 0.002 | +0.027 +- 0.008 | +0.16 +- 0.33 |
| absorbing floor and ceiling | moments | +0.025 +- 0.008 | -0.023 +- 0.032 | +0.60 +- 0.16 |
| | FDN | +0.015 +- 0.014 | +0.037 +- 0.025 | -0.37 +- 0.20 |
| corridor | moments | -0.026 +- 0.036 | -0.227 +- 0.085 | +0.54 +- 0.48 |
| | FDN | -0.071 +- 0.042 | +0.048 +- 0.024 | -2.19 +- 0.56 |

The full table (both late envelopes, every row) is in `summary.md`. In short:

- **Spatial.** With `"directional"` the moments reproduce the rays' anisotropy: the vertical dipole of the room with
  an absorbing floor and ceiling is 0.09 as in the rays (the FDN's 0.26), the corridor's cross dipoles 0.05 to 0.08
  as in the rays (the FDN's 0.20 to 0.24). The covariance error is lower than the FDN's in every room and order, most
  at third order (0.28 against 0.48 in the corridor), where the FDN's 96-direction grid leaks between channels.
- **"omni"** leaves the late field near isotropic in every room, as the FDN's omni mode does.
- **Energy and decay.** Over seeds T30 is within 0.03 s of the reference (the FDN's up to 0.07 s short in the
  corridor), EDT within 0.03 s in the office and the room with an absorbing floor and ceiling, and C50 0.5 to 0.65 dB
  high in every room (the FDN's 2.2 dB low in the corridor). The office's late level is 0.9 to 1.3 dB from the
  reference's late part, the FDN's 2.3 dB (its attenuate-only gains). The corridor's EDT is 0.23 +- 0.09 s short
  (the FDN's 0.05 s long): see Known limits.
- **Time.** 0.5 to 2.0 s per pair against 3 to 20 s: 4 to 7 times faster with "omni", 8 to 14 with "directional".

## Regression: HIL D 24.1

`tools/regression_reference.py` with `foa,mono,diffraction`, same seed, before (`d1da8db`) and after (this branch on
top of it, uncommitted at the time of recording). Recordings in `examples/results/reference/d1da8db` and
`d1da8db_sh_moments_final` (`compare.txt`, `against_mono.txt`).

- `mono`: 6/6 responses bit-identical.
- `foa` (`Ambisonics(reference=Mono(...))`, first order, scattering; energy from the rays) and `diffraction`
  (`Ambisonics(reference="mono")` with UDFA; network energy brought to the reference): every response changes, as the
  tail is a new noise; the largest difference -7.2 and -7.7 dB re peak.
- Tests: 1325 passed before, 1350 after (25 new), 2 skipped.
- Time: `foa` 443 -> 414 s, `diffraction` 306 -> 207 s (both include the mono trace).
- The fitted start after the rays (`d1da8db_sh_moments_holdfit` against `d1da8db_sh_moments_final`): `foa` changes by
  at most -72 dB re peak and its parameters by 0.01 at most (T30 against mono 0.022 -> 0.025 s); `diffraction`, whose
  energy is the network's, and `mono` are bit-identical. The table below is the fitted start's.
- The images' specular part with the rays' energy (`d1da8db_sh_moments_scatterfix`): `foa` runs with
  `scattering=True`, whose images kept their specular part already, and changes by at most -72.7 dB re peak;
  `diffraction` and `mono` are bit-identical.

Which recording is closer to the room's own `Mono` run, |hybrid - mono| averaged over the 6 pairs, overall values
(the largest single band in brackets), `examples/benchmarks/regression_against_mono.py`:

| | foa before | foa after | diffraction before | diffraction after |
| --- | --- | --- | --- | --- |
| T30 s | 0.057 (0.11) | **0.025** (0.16) | 0.057 (0.11) | **0.030** (0.08) |
| T20 s | 0.077 (0.27) | **0.048** (0.16) | 0.078 (0.28) | **0.040** (0.15) |
| EDT s | **0.047** (0.45) | 0.083 (0.46) | 0.052 (0.47) | **0.015** (0.14) |
| C50 dB | 1.31 (6.4) | **0.72** (11.1) | 1.01 (7.2) | **0.70** (5.3) |
| C80 dB | **0.49** (4.3) | 1.10 (7.4) | **0.35** (4.2) | 0.74 (2.4) |
| D50 % | 6.5 (34) | **3.7** (50) | 5.0 (39) | **3.8** (18) |
| Ts s | 0.008 | **0.003** | 0.007 | **0.003** |
| STI | 0.031 | **0.009** | 0.028 | **0.010** |
| G dB | **0.20** (3.1) | 0.34 (2.0) | 0.34 (3.2) | **0.28** (1.3) |

The FDN's late field was systematically too quiet early in its decay, so its clarity was biased up: mean signed C50
of +0.8 to +2.5 dB in every band in both methods. The moments remove that bias (`foa`: +0.0 to +0.95 dB above 125 Hz,
`diffraction`: -0.1 to -0.7 dB) and halve the T30, T20, Ts and STI errors.

What got worse:

- `foa`'s overall C80 (0.49 -> 1.10 dB), EDT (0.047 -> 0.083 s) and G (0.20 -> 0.34 dB). The per-band signed C80
  alternates in sign (+0.5 at 125 Hz, -0.7 at 1 kHz, +0.9 at 2 kHz) and averages near zero; it follows the ray
  histogram's expected energy where the mono response is one Poisson realisation of it.
- The largest single-band errors of `foa` at 125 Hz (C50 11 dB, C80 7 dB, D50 50 %): pair s2-r2, whose mono
  response has a C50 of -9.1 dB at 125 Hz against 0.3 to 4.5 dB in the other pairs. The FDN missed it by 6 dB, the
  moments by 11 dB; the mono run's 125 Hz octave is the sparse early part of one realisation.
- `diffraction`'s overall C80 (0.35 -> 0.74 dB); its largest band improved (4.2 -> 2.4 dB).

## Known limits

- The 63 Hz band of the late field is 3 dB below the reference's measured through the package's bank, and the 16 kHz
  band 2.7 dB above it, in every receiver of the office: pyroomacoustics' 512-point band filters cannot isolate the
  63 Hz octave (the reference's tail takes 125 Hz energy into it), and the reference's top band is filtered
  differently. Both are band-splitting differences of the reference, not of the rays' energy.
- A `Mono` reference object at the method's order now gives the rays' energy, the same responses as a dict the
  network's: the late fields differ (the early fields are bit-identical).
- The moments of a reference with few rays are as noisy as that reference; they are smoothed over 20 ms (energy) and
  100 ms (shape).
- Past the rays' -60 dB point the directional shape is held, where the FDN's lines keep decaying each at its own
  rate (the corridor's FDN turns fully axial after 1.5 s, the held moments stay at their last x/y/z of
  2.5/0.04/0.47). That part is 36 to 53 dB below the loudest 50 ms in the corridor; a decay per direction fitted from
  the rays would let the shape keep evolving.
- In the corridor the moments' EDT is 0.23 +- 0.09 s short of the reference over seeds 1 to 4 on v0.2.0 (the FDN's
  0.05 s long), where its T30 and C50 are closer than the FDN's. The corridor's decay is strongly bent (a fast early
  part, then the axial energy between the end walls). Not yet explained.
- On v0.2.0 (references traced with at least 20 000 rays) the EDT of the room with an absorbing floor and ceiling is
  -0.02 +- 0.03 s over seeds 1 to 4. The analysis below was made before, on `d1da8db`.
- In the room with an absorbing floor and ceiling the moments' EDT is short of the 2 s reference's: over seeds 42 and
  1 to 5 (12 responses) by 0.13 +- 0.03 s before the images kept their specular part, 0.09 +- 0.03 s after, the
  FDN's by +0.003 s. This room's EDT is fragile: its decay lingers between -5 and -10 dB for 300 ms, and the
  reference's EDT varies by 0.18 s between seeds. Splitting the reference into its image and ray parts:
  - The images: the hybrid's counted the early orders' scattered energy twice, fixed (above), a third of the bias.
  - The reference's ray part holds 10.5 % (+0.44 dB) more energy than its own rays from about 20 ms on.
    pyroomacoustics scales each impulse of its Poisson sequence by `1 / sqrt(local power)`, the power a two-bin
    rolling mean (`simulation/rt.py`, `directional_sequence`, `seq_rolling_power(..., filter_length_mult=2)`); a
    sequence asked for unit energy per bin delivers 1.10 to 1.11 at 144 and 1500 m^3. The moments deliver the rays'
    energy (1.001). The FDN follows the reference's measured envelope, excess included. Raising the moments by that
    0.44 dB leaves -0.06 +- 0.03 s: another third.
  - The rest (-0.06 +- 0.03 s, two standard errors; C50 +0.5 dB): the moments follow the reference's histogram
    within 0.3 dB but for the onset, which their 20 ms smoothing spreads towards the direct sound (+5 dB at 12 to
    25 ms, worth 0.006 s of EDT); likely the reference's sequence, whose energy per bin is lower while it is sparse
    (1.0 to 1.07 in its first 40 ms against 1.10 later). Not pursued.
