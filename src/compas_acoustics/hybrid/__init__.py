"""The DSP of :class:`compas_acoustics.Binaural` and :class:`compas_acoustics.Ambisonics`: image sources driving a directional FDN.

Needs pyFDN and spharpy, installed on Python >= 3.11.
"""

import soundfile as _soundfile

# flamo before 0.2.19 (via pyFDN) installs the abandoned PySoundFile 0.9.0 over soundfile (>= 0.12)
if tuple(int(part) for part in _soundfile.__version__.split(".")[:2]) < (0, 12):
    raise ImportError(
        "soundfile was replaced by the abandoned PySoundFile that flamo (via pyFDN) installs; "
        "repair it with: pip install --force-reinstall --no-deps soundfile"
    )

#: The hybrid-rir revision the hybrid modules were ported from.
UPSTREAM_REVISION = "a272da7"

__all__ = ["UPSTREAM_REVISION"]
