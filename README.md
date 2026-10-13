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
2. Unzip it and open `compas_acoustics_installer.gh` in Grasshopper.
3. Press **Install**. When the component says *Restart Rhino*, restart Rhino.

Installer options:

- `source`: path or download link of a `.whl` file. Empty: use the `.whl` file next to the `.gh` file.
- `venv`: name of the Rhino script environment to install into. Empty: `default`. With another name, the installed components use that environment.

Rhino 8 uses Python 3.9. With Python older than 3.12, the Speech Transmission Index (STI) is not available.

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
