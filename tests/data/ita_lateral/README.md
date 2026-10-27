# ITA Toolbox lateral energy fraction oracle

`ita_lf_cases.json` holds LF (J_LF) and LFC (J_LFC) per octave band, 125 Hz to 8 kHz, computed by the ITA Toolbox's
`ita_roomacoustics_lateral` (ISO 3382-1) with `ita_lf_cases.m` in MATLAB R2025b on 2026-10-23 (the toolbox as
installed then; `ita_toolbox_version_number` reported nothing). They check `Analyser.lateral_fraction`.

Signals, 48 kHz, 1 s: an omnidirectional channel and a figure-of-eight channel whose null faces the direct sound.
The direct sound arrives from 2 m; one reflection of amplitude 0.5 follows 20 ms later, lateral (`lateral`) or at
45 degrees (`oblique`); in `outside` the lateral reflections arrive 3 ms and 90 ms after the direct sound only.

Only the ratios between cases are a reference: oblique over lateral is 0.500 for LF (cos^2) and 0.707 for LFC (cos)
in every band. The absolute values are high: `ita_roomacoustics_lateral` moves the direct sound to t = 0
(`ita_time_shift`, -20 dB) and then filters zero-phase, so the half of the direct sound's band response before t = 0
falls outside the 0-80 ms window (51 % of its energy is kept at 125 Hz, 56 % at 1 kHz, 71 % at 8 kHz). With 56 %
the 1 kHz lateral case is 0.25 / (0.561 + 0.25) = 0.308, as ITA gives; the analytical value is 0.25 / 1.25 = 0.2.
The analyser cuts its windows from the broadband response before filtering and keeps the whole direct sound.
