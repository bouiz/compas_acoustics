# Install in Python

`compas_acoustics` needs Python 3.9 or newer; the `Binaural` and `Ambisonics` methods need 3.11 or newer.
Create an environment, for example with conda:

```bash
conda create -n acoustics python=3.12
conda activate acoustics
```

Then install it from PyPI:

```bash
pip install compas_acoustics
```

To install the latest `main` instead, install it from the git repository:

```bash
pip install "compas_acoustics @ git+https://github.com/gramaziokohler/compas_acoustics.git"
```

## Binaural and ambisonics methods

The `Binaural` and `Ambisonics` methods share one hybrid engine (image sources, edge diffraction and a
directional feedback delay network, after [hybrid-rir](https://github.com/gpapadimitri/hybrid-rir)) and differ only
in how they render it. They need pyFDN, spharpy and multislope, which `compas_acoustics` installs on Python 3.11
and newer; on Python 3.9 and 3.10 (Rhino 8's Python) it leaves them out and the methods are not available.

pyFDN brings flamo and with it PyTorch: about 0.5 GB on macOS, several GB with CUDA on Linux. Only pyFDN's training
modules use it; the methods run without them.

## Check the installation

```bash
python -c "import compas_acoustics; print(compas_acoustics.__version__)"
```

Then follow the [tutorial](../python/tutorial.md).

## Development

To work on `compas_acoustics`, install it from a local clone with the "dev" requirements:

```bash
git clone https://github.com/gramaziokohler/compas_acoustics.git
cd compas_acoustics
pip install -e ".[dev]"
```

See [Contributing to the documentation](../contributing/docs.md) for building the documentation.
