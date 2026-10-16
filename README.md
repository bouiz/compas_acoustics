[![Made with COMPAS](https://compas.dev/badge.svg)](https://compas.dev)

# compas_acoustics



## Installation

Create a virtual environment with `python >= 3.10`
```bash
conda create -n your_env_name python=3.10
```

To install the latest version for development, do:

```bash
git clone https://github.com/gramaziokohler/compas_acoustics.git
cd compas_acoustics
conda activate your_env_name
pip install -e ".[dev]"
```

## Install in Rhino 8 / Grasshopper

`compas_acoustics` is not on PyPI yet. To install it in Rhino 8, use the Grasshopper installer:

1. Run `invoke installer-bundle`. It creates `dist/compas_acoustics_installer.zip`.
   GitHub Actions also builds it: download it from the `compas_acoustics_installer` artifact of the build workflow, or from the release.
2. Unzip it and open `compas_acoustics_installer.gh` in Grasshopper.
3. Press **Install**. When the component says *Restart Rhino*, restart Rhino.

Installer options:

- `source`: path or download link of a `.whl` file. Empty: use the `.whl` file next to the `.gh` file.
- `venv`: name of the Rhino script environment to install into. Empty: `default`. With another name, the installed components use that environment.

Rhino 8 uses Python 3.9. With Python older than 3.12, the Speech Transmission Index (STI) is not available.

## Getting started

Every simulation method runs through `scene.simulate`:

```python
import compas_acoustics as ca

scene = ca.Scene("room")
scene.add_acoustic_object(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
scene.add_source(ca.Source("talker", 2, 2, 1.6))
scene.add_receiver(ca.Receiver("listener", 6, 4, 1.2, direction=[-1, -0.5, 0]))

irs = scene.simulate(ca.Pyroomacoustics(ray_tracing=True))   # image sources + ray tracing
```

The notebook [`examples/notebooks/01_pyroomacoustics_simulation.ipynb`](examples/notebooks/01_pyroomacoustics_simulation.ipynb)
walks through simulation, analysis, export and auralisation. The Grasshopper components are in
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
