"""The octave band grid of the hybrid synthesis.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ 8a70661, src/room.py).
"""

from __future__ import annotations

import numpy as np
import pyFDN

from compas_acoustics.room_geometry import SYNTHESIS_BANDS

# The package-wide octave band grid, fixed at pyFDN.octave_bands() defaults so
# every band-indexed array shares one axis with pyFDN.estimate_rt_bands and
# pyFDN.estimate_initial_level_bands.
BAND_EDGES, BANDS = pyFDN.octave_bands()
N_BANDS = BANDS.size

# Needed wherever a spectral *density* has to be expressed in pyFDN's per-band
# ``level`` units: the estimator matches total energy inside each band, so a
# band's share of a spectrally flat signal grows with its width and the level
# of a white decay rises as sqrt(bandwidth).
BANDWIDTHS = BAND_EDGES[:, 1] - BAND_EDGES[:, 0]


def check_bands(bands: np.ndarray) -> None:
    """Raise if ``bands`` is not the package band grid.

    Silently accepting a different grid would misalign every band-indexed array
    in the package, so this is checked at every boundary where a caller supplies
    band centre frequencies.
    """
    bands = np.asarray(bands, dtype=float)
    if bands.shape != BANDS.shape or not np.allclose(bands, BANDS):
        raise ValueError(f"band grid must be {BANDS.tolist()} Hz, got {bands.tolist()}")


# RoomGeometry puts materials on SYNTHESIS_BANDS without importing pyFDN;
# the two must be one grid.
check_bands(np.asarray(SYNTHESIS_BANDS))
