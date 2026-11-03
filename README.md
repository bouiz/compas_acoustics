[![Made with COMPAS](https://compas.dev/badge.svg)](https://compas.dev)

# compas_acoustics

compas_acoustics simulates, analyses and auralises the acoustics of rooms modelled with COMPAS or in Rhino and Grasshopper. It computes impulse responses with the image source method and ray tracing (Mono), and with a hybrid engine of image sources, edge diffraction and a directional feedback delay network (Binaural, Ambisonics); it derives room acoustic parameters (ISO 3382) from them and convolves them with anechoic recordings to listen to a design.

## Features

- **Simulation methods**: `Mono` (image sources and ray tracing, with pyroomacoustics), `Binaural` and `Ambisonics` (a hybrid of image sources, edge diffraction and a directional feedback delay network), all through `scene.simulate`.
- **Analysis**: reverberation times, clarity and definition, direct-to-reverberant ratio, lateral fraction, stage support, strength and STI from an impulse response (`Analyser`).
- **Auralisation**: convolve mono, binaural or ambisonic responses with anechoic recordings (`Auraliser`), and play them (`AudioPlayer`).
- **Materials library**: 95 absorbing and scattering materials in categories, searchable from Python and Grasshopper.
- **Grasshopper plugin**: components for Rhino 8 on Windows and macOS that run every computation in a Python worker.

## Installation

`compas_acoustics` is not on PyPI yet. The [installation guide](https://gramaziokohler.github.io/compas_acoustics/latest/installation/index.html) has the details.

**Rhino 8 / Grasshopper** (Windows and macOS): download `compas_acoustics_installer.zip` from the
[latest release](https://github.com/gramaziokohler/compas_acoustics/releases/latest), unzip it, open
`compas_acoustics_installer.gh` in Grasshopper, press **Install**, and restart Rhino. The installer adds the plugin and
creates the worker, the Python 3.12 environment the components compute in.

**Python** (3.9 or newer; the `Binaural` and `Ambisonics` methods need 3.11 or newer):

```bash
pip install "compas_acoustics @ git+https://github.com/gramaziokohler/compas_acoustics.git"
```

Add a release tag (`…compas_acoustics.git@v0.1.0`) to install that release. For development:

```bash
git clone https://github.com/gramaziokohler/compas_acoustics.git
cd compas_acoustics
pip install -e ".[dev]"
```

## Getting started

Every simulation method runs through `scene.simulate`:

```python
import compas_acoustics as ca

scene = ca.Scene("room")
scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
scene.add_source(ca.Source("talker", 2, 2, 1.6))
scene.add_receiver(ca.Receiver("listener", 6, 4, 1.2, direction=[-1, -0.5, 0]))

mono = scene.simulate(ca.Mono(ray_tracing=True))              # pyroomacoustics: image sources + ray tracing
binaural = scene.simulate(ca.Binaural())                       # needs Python >= 3.11
ambisonics = scene.simulate(ca.Ambisonics(sh_order=1))         # ACN/SN3D, for a decoder elsewhere

speech = ca.Defaults().ANECHOIC_SOUNDFILES_LIBRARY_PATH + "/female_speech.wav"
signal = ca.Auraliser("room", list(binaural.values()), anechoic_sounds=[speech]).convolve()   # (samples, 2)
```

`Auraliser` convolves and keeps the format of its responses: mono, binaural, or ambisonic ACN/SN3D (AmbiX) to save as a
multichannel file for an ambisonic decoder elsewhere. There is no ambisonic-to-binaural or stereo decoder: for headphones,
simulate with `Binaural`.

`examples/` has short scripts: `simple_simulation_shoebox.py` (a room from scratch), `obj.py` (a room from OBJ files), `rays.py`, `scattered_rays.py` and `specular_reflections.py` (ray and image-source paths for display).

## Grasshopper plugin development

The plugin is `src/compas_acoustics_ghcsharp/CompasAcoustics.sln`: build it with `invoke build-gha` (or `dotnet build`), test it with
`dotnet test src/compas_acoustics_ghcsharp/CompasAcoustics.sln`.
Each component takes its names, descriptions, ports and icon from `src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/Specs/<CA_name>/`.
After a change to `src/compas_acoustics/pb/protobuf_defs/.../acoustics.proto`, run `invoke generate` (Python and C# bindings)
and `python tests/pb_fixtures/generate.py` (the payloads the C# tests read).

## Materials library

Materials are grouped in categories (`MaterialCategory`). To update the materials that come from pyroomacoustics, run:

```bash
python tools/import_backend_materials.py
```

## Documentation

For further "getting started" instructions, a tutorial, examples, and an API reference,
please check out the online documentation here: [compas_acoustics docs](https://gramaziokohler.github.io/compas_acoustics):
concepts, installation, a Python and a Grasshopper track with examples, a reference of every Grasshopper component, and
the API reference.

## Issue Tracker

If you find a bug or if you have a problem with running the code, please file an issue on the [Issue Tracker](https://github.com/gramaziokohler/compas_acoustics/issues).
