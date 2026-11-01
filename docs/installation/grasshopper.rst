********************************************************************************
Install in Rhino 8 and Grasshopper
********************************************************************************


The Grasshopper components are a compiled plugin for Rhino 8 on Windows and macOS.

1. Download ``compas_acoustics_installer.zip`` from the
   `latest release <https://github.com/gramaziokohler/compas_acoustics/releases/latest>`_.
2. Unzip it and open ``compas_acoustics_installer.gh`` in Grasshopper.
3. Press **Install**. When the component says *Restart Rhino*, restart Rhino.

The installer copies the plugin into Grasshopper's Libraries folder and creates
the worker, a Python 3.12 environment with ``compas_acoustics`` in which the
components run every computation. The first installation downloads PyTorch,
several GB.
