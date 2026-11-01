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

Create a virtual environment with `python >= 3.9` (Rhino 8 runs 3.9; the hybrid method needs 3.11 or newer)
```bash
conda create -n your_env_name python=3.12
```

To install it in Python from the repository (it is not on PyPI yet):

```bash
pip install "compas_acoustics @ git+https://github.com/gramaziokohler/compas_acoustics.git"
```

This installs the latest `main`. To install a specific release, add its tag, for example `…compas_acoustics.git@v0.1.0`; the tags are on the [releases page](https://github.com/gramaziokohler/compas_acoustics/releases).

To install the latest version for development, do:

```bash
git clone https://github.com/gramaziokohler/compas_acoustics.git
cd compas_acoustics
conda activate your_env_name
pip install -e ".[dev]"
```

### Binaural and ambisonics methods (Python >= 3.11)

The `Binaural` and `Ambisonics` simulation methods share one hybrid engine (image sources, edge
diffraction and a directional feedback delay network) and differ only in how they render it.
They need pyFDN, spharpy and multislope, which `compas_acoustics` installs on Python 3.11 and
newer; on Python 3.9 and 3.10 (Rhino 8) it leaves them out and the methods are not available.
pyFDN brings flamo and with it PyTorch (about 0.5 GB on macOS, several GB with CUDA on Linux),
which only pyFDN's training modules use; the methods run without them.

## Install in Rhino 8 / Grasshopper

The Grasshopper components are a compiled plugin, `CompasAcoustics.gha` (C#, `src/compas_acoustics_ghcsharp/`), for Rhino 8 on
Windows and macOS. Everything they compute runs in the worker, a Python 3.12 with `compas_acoustics`; the plugin
sends it scenes, responses and signals as [compas_pb](https://github.com/compas-dev/compas_pb) protobuf messages.

`compas_acoustics` is not on PyPI yet. To install it in Rhino 8, use the Grasshopper installer:

1. Download `compas_acoustics_installer.zip` from the [latest release](https://github.com/gramaziokohler/compas_acoustics/releases/latest):
   the wheel, the plugin (`gha/`) and `compas_acoustics_installer.gh`.
   To build it yourself, run `invoke installer-bundle` (it needs the .NET 8 SDK); it creates `dist/compas_acoustics_installer.zip`.
2. Unzip it and open `compas_acoustics_installer.gh` in Grasshopper.
3. Press **Install**. When the component says *Restart Rhino*, restart Rhino.

The installer copies the plugin into Grasshopper's Libraries folder (`compas_acoustics/`), removes the CPython
components of earlier versions, and creates the worker: a Python 3.12 environment made with `uv` (it installs
PyTorch, several GB, the first time), recorded in `~/.compas_acoustics/worker.json`. Set the environment variable
`COMPAS_ACOUSTICS_WORKER_PYTHON` to use another interpreter that has this version of `compas_acoustics`.

Installer options:

- `source`: path of the `.whl` file in the bundle. Empty: use the `.whl` file next to the `.gh` file. The plugin must be next to it, in `gha/`.
- `venv`: name of the Rhino script environment that gets `compas_acoustics` too, for your own scripts. Empty: `default`.
- `worker`: ignored; the worker is always set up.

### The worker

The worker starts with the first component that needs it (a few seconds; half a minute the first time) and stops with
Rhino. Every Rhino session starts its own, on a free port of `127.0.0.1`, and shares a random token with it: it answers
`POST /api/<function>` of `compas_acoustics.worker.api` only with the token, takes and returns `compas_pb` payloads, and exits
when Rhino ends or dies. It must run the same `compas_acoustics` version and protobuf schema as the plugin; otherwise the
components say to run the installer again. A Python script can use the same worker through `compas_acoustics.worker.client()`.

### Using the components

Choices sit on the component bodies (switches, dropdowns, buttons); a wire into the same input wins over the control. The long
runs (simulations, analysis, auralisation, rays) work in the background: the canvas stays responsive, the message shows the
time and the worker's stage, and Cancel stops the run. Material Library opens a searchable picker on a click of its body;
Plot 2D draws its charts on the canvas; the Audio Player shows the signal (an auralisation, a file, or an impulse response from Deconstruct IR Graph) and plays, pauses, seeks and sets the volume. The
Worker component (Util) shows the worker's state and opens its log; every component's right-click menu has "Open worker log".

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

`examples/` has short scripts: `shoebox.py` (a room from scratch), `obj.py` (a room from OBJ files), `rays.py`, `scattered_rays.py` and `specular_reflections.py` (ray and image-source paths for display).

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
please check out the online documentation here: [compas_acoustics docs](https://gramaziokohler.github.io/compas_acoustics)

## Issue Tracker

If you find a bug or if you have a problem with running the code, please file an issue on the [Issue Tracker](https://github.com/gramaziokohler/compas_acoustics/issues).
