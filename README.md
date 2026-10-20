[![Made with COMPAS](https://compas.dev/badge.svg)](https://compas.dev)

# compas_acoustics



## Installation

Create a virtual environment with `python >= 3.9` (Rhino 8 runs 3.9; the hybrid method needs 3.11 or newer)
```bash
conda create -n your_env_name python=3.12
```

To install the latest version for development, do:

```bash
git clone https://github.com/gramaziokohler/compas_acoustics.git
cd compas_acoustics
conda activate your_env_name
pip install -e ".[dev]"
```

### Hybrid ambisonic method (Python >= 3.11)

The `Hybrid` simulation method (image sources, edge diffraction and a directional
feedback delay network, ported from [hybrid-rir](https://github.com/gpapadimitri/hybrid-rir))
needs pyFDN, spharpy and multislope, which `compas_acoustics` installs on Python 3.11 and
newer; on Python 3.9 and 3.10 (Rhino 8) it leaves them out and the method is not available.
pyFDN brings flamo and with it PyTorch (about 0.5 GB on macOS, several GB with CUDA on Linux),
which only pyFDN's training modules use; the hybrid method runs without them.

## Install in Rhino 8 / Grasshopper

`compas_acoustics` is not on PyPI yet. To install it in Rhino 8, use the Grasshopper installer:

1. Run `invoke installer-bundle`. It creates `dist/compas_acoustics_installer.zip`.
   GitHub Actions also builds it: download it from the `compas_acoustics_installer` artifact of the build workflow, or from the release.
2. Unzip it and open `compas_acoustics_installer.gh` in Grasshopper.
3. Press **Install**. When the component says *Restart Rhino*, restart Rhino.

Installer options:

- `source`: path or download link of a `.whl` file. Empty: use the `.whl` file next to the `.gh` file.
- `venv`: name of the Rhino script environment to install into. Empty: `default`. With another name, the installed components use that environment.
- `worker`: empty, `binaural` or `hybrid`. See below.

Rhino 8 uses Python 3.9. With Python older than 3.11, the Speech Transmission Index (STI) and sound strength (G) are not available there: with a worker (below), the Analyse IR component computes them in the worker; without one, it leaves them out.

### Python >= 3.11 features in Rhino 8: the worker

The hybrid method and binaural decoding need Python >= 3.11. Rhino 8 runs
Python 3.9. With the installer's `worker` input set to `binaural` or `hybrid` (both install the same packages, PyTorch included; `hybrid` also checks that the hybrid method imports),
the installer creates a second Python 3.12 environment with `uv` and records it in `~/.compas_acoustics/worker.json`. Set the environment variable
`COMPAS_ACOUSTICS_WORKER_PYTHON` to use another interpreter that has `compas_acoustics`.

Simulate Hybrid, Auralise and Analyse IR (for STI and G) then run those parts in the worker, a local [`compas.rpc`](https://compas.dev) server that
starts on first use (a few seconds) and exchanges scenes as COMPAS JSON and audio as WAV files. Every Rhino session starts its own worker, on a free
port of `127.0.0.1`, and shares a random token with it; the worker serves only `compas_acoustics.worker.api`, refuses calls without the token, reads and
writes only in temporary folders made for the call, and exits when the session ends or dies. It checks that it runs the same code as Rhino's Python, and tells you
to run the installer again if not. Without a worker nothing changes: the components run in Rhino's Python and report what is missing.

## Getting started

Every simulation method runs through `scene.simulate`:

```python
import compas_acoustics as ca

scene = ca.Scene("room")
scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
scene.add_source(ca.Source("talker", 2, 2, 1.6))
scene.add_receiver(ca.Receiver("listener", 6, 4, 1.2, direction=[-1, -0.5, 0]))

mono = scene.simulate(ca.Pyroomacoustics(ray_tracing=True))   # image sources + ray tracing
ambisonic = scene.simulate(ca.Hybrid())                        # first-order ambisonics; needs Python >= 3.11
```

`examples/office_hybrid.py` runs both methods on an office, with reverberation times and an auralisation. The Grasshopper components are in
`src/compas_acoustics_ghpython/components`; rebuild their user objects with `invoke build-cpython-ghuser-components`.

## Materials library

Materials are grouped in categories (`MaterialCategory`). To update the materials that come from pyroomacoustics, run:

```bash
python tools/import_backend_materials.py
```

## Documentation

For further "getting started" instructions, a tutorial, examples, and an API reference,
please check out the online documentation here: [compas_acoustics docs](https://gramaziokohler.github.io/compas_acoustics)

## Issue Tracker

If you find a bug or if you have a problem with running the code, please file an issue on the [Issue Tracker](https://github.com/gramaziokohler/compas_acoustics/issues).
