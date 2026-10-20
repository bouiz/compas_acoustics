# RAZR surface scattering oracle

`razr_scattering_oracle.json` holds numbers derived from RAZR 0.96.3 (`~/Documents/TOOLBOX/RAZR/build_0963`; its
`get_razr_version` reports `0.96`) run as a black box by `razr_scattering_oracle.m` in MATLAB R2025b, on 2026-10-20.
They check the surface scattering of the hybrid (Ewert et al. 2025, Acta Acustica 9, 9) in `tests/hybrid/test_scattering.py`.

Options: `fs` 48000, `spat_mode='diotic'`, `ism_order=3`, `ism_jitter_factor=0`, `enableSR=[0 0]`,
`ism_enableAirAbsFilt=0`, `fdn_enable_apc=0`, every other option RAZR's default; scattering on
(`ism_enable_scattering=1`) against off. `room.scat_amount` d in {0.25, 0.5} on all six walls, `room.scat_cutoff` 20 Hz,
so `|H_d|^2 = d` over the audio band.

Rooms (the shoeboxes of `validation/2026-10-16-hybrid-transition/scripts/razr/razr_exact_rooms.m`), `room.materials`
the absorption on every wall and band, receiver looking along +x:

| Room | Box (m) | Absorption | Source (m) | Receiver (m) |
|---|---|---|---|---|
| 1 | 10 x 8 x 4 | 0.15 | 2.5, 3.0, 1.5 | 7.5, 5.5, 1.2 |
| 2 | 6 x 5 x 3 | 0.10 | 1.5, 1.5, 1.5 | 4.5, 3.5, 1.2 |
| 3 | 6 x 5 x 3 | 0.20 | 4.0, 3.5, 1.7 | 2.0, 2.0, 1.5 |
| 4 | 8 x 6 x 3.5 | 0.10 | 2.2, 1.7, 1.4 | 5.9, 4.1, 1.3 |

Per room and d: `specular_ratio[k]`, the energy of the order-k images' specular signals (`early_refl_sigmat`) with
scattering on over off, and `scattered_ratio[k]`, the energy of their scattered signals (`early_refl_sigmat_diffuse`)
over the specular ones with scattering off, both above 5 kHz (fourth-order Butterworth high-pass), k = 1, 2, 3;
`d_early`, `d_late`, `d_total` and `d_direct_early`, the energy change of `sig_early`, `sig_late`, `sig` and
`sig_direct + sig_early` with scattering on against off, in dB.

Licence: RAZR is licensed CC BY-NC-ND 4.0. No RAZR code is stored here, only the numbers derived from its output and the
script that calls it.
