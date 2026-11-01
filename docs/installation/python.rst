********************************************************************************
Install in Python
********************************************************************************


Create an environment with Python 3.9 or newer, then install it from the git
repository:

.. code-block:: bash

    pip install "compas_acoustics @ git+https://github.com/gramaziokohler/compas_acoustics.git"

This installs the latest ``main``. To install a specific release, add its tag,
for example ``…compas_acoustics.git@v0.1.0``; the tags are on the
`releases page <https://github.com/gramaziokohler/compas_acoustics/releases>`_.

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
