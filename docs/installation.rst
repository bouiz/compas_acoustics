********************************************************************************
Installation
********************************************************************************

``compas_acoustics`` is not on PyPI yet. Install it in Rhino with the installer
bundle of a GitHub release, or in Python from the git repository.


Rhino 8 and Grasshopper
=======================

The Grasshopper components are a compiled plugin for Rhino 8 on Windows and macOS.

1. Download ``compas_acoustics_installer.zip`` from the
   `latest release <https://github.com/gramaziokohler/compas_acoustics/releases/latest>`_.
2. Unzip it and open ``compas_acoustics_installer.gh`` in Grasshopper.
3. Press **Install**. When the component says *Restart Rhino*, restart Rhino.

The installer copies the plugin into Grasshopper's Libraries folder and creates
the worker, a Python 3.12 environment with ``compas_acoustics`` in which the
components run every computation. The first installation downloads PyTorch,
several GB.


Python
======

Create an environment with Python 3.9 or newer, then install a release from
the git repository:

.. code-block:: bash

    pip install "compas_acoustics @ git+https://github.com/gramaziokohler/compas_acoustics.git@v0.1.0"

The ``Binaural`` and ``Ambisonics`` methods need Python 3.11 or newer: there,
``compas_acoustics`` also installs pyFDN, spharpy and multislope. On Python 3.9
and 3.10 (Rhino 8's Python) these methods are not available.


Development
===========

To work on ``compas_acoustics``, install it from a local clone with the "dev"
requirements:

.. code-block:: bash

    git clone https://github.com/gramaziokohler/compas_acoustics.git
    cd compas_acoustics
    pip install -e ".[dev]"
