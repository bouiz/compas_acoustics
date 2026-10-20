# RAZR surface scattering oracle

`razr_scattering_oracle.json` holds numbers derived from RAZR 0.96.3 (`build_0963`; its `get_razr_version` reports
`0.96`). They were made on 2026-10-20 by running RAZR as a black box with `razr_scattering_oracle.m` in MATLAB R2025b:
`razr_scattering_oracle('razr_scattering_oracle.json', '/path/to/RAZR/build_0963')`. They check that the hybrid's surface
scattering (Ewert et al. 2025, Acta Acustica 9, 9) implements RAZR's method. RAZR's levels are not a reference.

Options: `fs` 48000, `spat_mode='diotic'`, `ism_order=3`, `ism_jitter_factor=0`, `enableSR=[0 0]`,
`ism_enableAirAbsFilt=0`, `fdn_enable_apc=0`, every other option RAZR's default. Scattering on
(`ism_enable_scattering=1`) is compared against off. `room.scat_cutoff` is 20 Hz, so `|H_d|^2 = d` over the audio band.
Walls are ordered `[-z -y -x +x +y +z]`, as in RAZR's `scat_amount`.

Rooms (the shoeboxes of `validation/2026-10-16-hybrid-transition/scripts/razr/razr_exact_rooms.m`), with `room.materials`
the absorption on every wall and band, and the receiver looking along +x:

| Room | Box (m) | Absorption | Source (m) | Receiver (m) |
|---|---|---|---|---|
| 1 | 10 x 8 x 4 | 0.15 | 2.5, 3.0, 1.5 | 7.5, 5.5, 1.2 |
| 2 | 6 x 5 x 3 | 0.10 | 1.5, 1.5, 1.5 | 4.5, 3.5, 1.2 |
| 3 | 6 x 5 x 3 | 0.20 | 4.0, 3.5, 1.7 | 2.0, 2.0, 1.5 |
| 4 | 8 x 6 x 3.5 | 0.10 | 2.2, 1.7, 1.4 | 5.9, 4.1, 1.3 |

## `cases`: per room, the same d on every wall, d in {0.25, 0.5}

- `specular_ratio[k]`: the energy of the order-k images' specular signals (`early_refl_sigmat`) with scattering on,
  over off.
- `scattered_ratio[k]`: the energy of their scattered signals (`early_refl_sigmat_diffuse`), over the specular ones with
  scattering off.
- Both ratios are taken above 5 kHz (fourth-order Butterworth high-pass), for k = 1, 2, 3. RAZR gives `(1 - d)^k` and
  `d^k`. So the scattered part of an image is the product of `H_d` over all its reflections, not the part scattered at
  its last reflection. At d = 0.5 the two agree; at d = 0.25 they do not (0.0625 against 0.1875 at k = 2).
- `mean_distance`: the receiver's mean distance to the six walls, RAZR's R for the time spread (`get_ism_setup.m`).
- `allpass_delays`: the four delays, in samples, of RAZR's time-spread cascade (`ism_setup.a_tspr`), read off the
  denominator's nonzero taps.
- `d_early`, `d_late`, `d_total`, `d_direct_early`: the energy change of `sig_early`, `sig_late`, `sig` and
  `sig_direct + sig_early` with scattering on against off, in dB. **Informational only, not tested**: RAZR's tail
  follows Eyring and is not a reference for compas' levels.

## `per_wall`: room 1, d = [0.1 0.2 0.3 0.4 0.5 0.6]

For each image (`order`, `pattern`, `positions` in room coordinates), its own `specular_ratio` and `scattered_ratio`,
defined as above. They are the products of `1 - d_i` and of `d_i` over the image's reflections, within 1e-5.

## `coupling`: how the scattered parts enter RAZR's FDN, room 1 with the per-wall d

RAZR builds the FDN input in `gen_fdn_input` (p-code). The script runs it through `create_rir.m`'s own steps, because
`op.debugMode`, which would store the FDN input, fails in `getReflFiltCoeff.m:56`.

For each FDN channel, the input is solved by least squares as a sum of two kinds of signal:
- the last-order specular signals (`curr_sigmat`), placed `num_pre_zeros` samples late;
- the scattered signals of every reflected image, smeared by `ism_setup.b_tspr / a_tspr` and placed
  `num_pre_zeros - delays(ch) + 1` samples late, so that they skip the channel's delay.

Fields:
- `residual`: the unexplained fraction of the input energy. `residual_unsmeared`: the same, with unsmeared scattered
  signals.
- `specular_gains` (last-order images x channels) and `scattered_gains` (all images x channels, zero for the direct
  sound): the solved weights.
- `azim`, `elev`: the channel directions. `relpos`: the images' positions relative to the receiver.

The solve for each mapping:
- `fdn_smart_ch_mapping = 0` (RAZR's default): residual 1e-29. Image j's signal enters channel `j mod 24` with weight 1,
  for the specular and the scattered parts alike. The mapping goes by index, not by direction.
- `fdn_smart_ch_mapping = 1`: residual 1e-29. The scattered part of every image of orders 1 to 3, the last order
  included, enters with weights `w = clip(u . v, 0) / ||clip(u . v, 0)||_2`. Here `v` is the image's direction of
  arrival and `u` the channel direction. The last order's specular part enters the same way with `-v`. Both are
  unit-norm rows, so a scattered part and the specular drive enter with equal weight. The weights follow the image's
  direction, not the normal of the reflecting wall.

## `line_filter`: is a scattered part heard through the line's absorption filter?

The same room, run through RAZR's FDN (`feedback_delay_network`) with `fdn_fmatrix = 'eye'` and `fdn_enableReflFilt = 0`.
Each channel in turn gets one impulse, placed as a scattered part is placed (`num_pre_zeros - delays(ch) + 1 + 200`) or
as a specular drive signal is (`num_pre_zeros + 200`). The first `n = 400` samples the channel outputs are compared with
the unfiltered impulse and with the line's absorption filter (`fdn_setup.b_abs / a_abs`).

- `max_error_vs_absorption_filter`: about 2e-13 for both kinds of input.
- `error_vs_impulse`: 0.08 to 0.32 for both.

So a scattered part passes the line's absorption filter once before it is heard, exactly as a drive signal does. It
differs from the drive only in skipping the delay. `heard_at` gives the output sample relative to the input's arrival
sample: 1 for every scattered input (heard one sample after its arrival), and the channel's delay for the specular input.

Licence: RAZR is licensed CC BY-NC-ND 4.0. No RAZR code is stored here, only the numbers derived from its output and the
script that calls it.
