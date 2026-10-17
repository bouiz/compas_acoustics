"""Hybrid ambisonic room impulse responses: image sources driving a directional FDN.

Ported from hybrid-rir (gpapadimitri/hybrid-rir); every generated module names
the upstream revision, :data:`UPSTREAM_REVISION`. Needs the ``hybrid`` extra
(pyFDN, spharpy). Used through :class:`compas_acoustics.simulation_methods.Hybrid`, a
simulation method (:mod:`compas_acoustics.simulation_methods`) for :meth:`compas_acoustics.Scene.simulate`;
this package holds the DSP it is built on.
"""

import soundfile as _soundfile

# flamo (via pyFDN) installs the abandoned PySoundFile, which overwrites soundfile.
# PySoundFile is 0.9.0 and also defines __libsndfile_version__; the maintained package is >= 0.12.
if tuple(int(part) for part in _soundfile.__version__.split(".")[:2]) < (0, 12):
    raise ImportError(
        "soundfile was replaced by the abandoned PySoundFile that flamo (via pyFDN) installs; "
        "repair it with: pip install --force-reinstall --no-deps soundfile"
    )

#: The hybrid-rir revision the modules of this package were ported from (tests/hybrid/UPSTREAM_REVISION).
UPSTREAM_REVISION = "a272da7"

__all__ = ["UPSTREAM_REVISION"]
