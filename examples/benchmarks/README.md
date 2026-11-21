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
