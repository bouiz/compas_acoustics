# Benchmarks

Scenes and scripts for checking the engine on real rooms. They are not rendered as documentation examples (the docs
build reads `examples/*.py` only).

## HIL D 24.1

`hil_d24_1.obj` is a Rhino export (Y-up) of lecture room HIL D 24.1 (ETH Hönggerberg), with one group per
`Acoustics::<material>` layer and the sources and receivers as point layers. `hil_d24_1.py` loads it
(`obj_scene_utils.py`), simulates every source-receiver pair with `Mono` and with `Ambisonics` (UDFA edge diffraction,
surface scattering, the mono run as reference), and writes tables, CSVs and figures (`results_plots.py`) to
`examples/benchmarks/results/hil_d24_1/`:

```bash
python examples/benchmarks/hil_d24_1.py
```

## Regression recordings

`tools/regression_reference.py` records the example's responses, their room parameters and the test suite's outcome
for the checked-out commit, and compares two recordings:

```bash
python tools/regression_reference.py record examples/results/reference/<commit>
python tools/regression_reference.py record examples/results/reference/<commit> --methods foa,mono,diffraction
python tools/regression_reference.py compare examples/results/reference/<a> examples/results/reference/<b>
```

`tools/prepared_geometry.py` does the same for the output of `compas_acoustics.geometry.prepare_mesh` on a corpus of
scenes, face by face.

## The ambisonic late field: SH-energy moments against the FDN

`sh_moment_late_field.py` simulates three shoeboxes of increasing anisotropy (a diffuse office, a room with an
absorbing floor and ceiling, a corridor with absorbing side walls) with `Ambisonics` at first and third order, with
`late_field="moments"` and `"fdn"` and both late envelopes, all on the same rays (seed 42). It compares the W
channel's room parameters and late level with the reference, the late field's channel covariance, diffuseness and
dipole energies with those of the rays, and the time spent on the late field, and writes `results.json` and
`summary.md` to `examples/results/sh_moments/` (about 15 minutes):

```bash
python examples/benchmarks/sh_moment_late_field.py
python examples/benchmarks/sh_moment_late_field.py --orders 1 --cases corridor
```

The method and the results are in `docs/validation/sh_moment_late_field.md`.

`sh_moment_plots.py` draws explanatory figures for the same rooms (first receiver, one octave band, both late
fields with the directional envelope) into `examples/results/sh_moments/plots/` (about a minute): each room with
its absorption and the early arrivals at the receiver; maps over azimuth and elevation of where the late energy
arrives from in three time windows (the rays per grid direction, the rays as an order-N signal sees them, and the
steered power of the moments' and the FDN's late fields); the dipole energies over time; the moment grid against the
96 FDN lines:

```bash
python examples/benchmarks/sh_moment_plots.py
python examples/benchmarks/sh_moment_plots.py --cases corridor --order 1 --band 500
```

`regression_against_mono.py` reads two regression recordings (the `foa`, `diffraction` and `mono` JSON files of a
recording directory) and says, per room parameter, which of them is closer to its own mono run.
