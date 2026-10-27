"""The octave band grid of the hybrid synthesis, shared with the decay estimators."""

from __future__ import annotations

import numpy as np
import pyFDN

from compas_acoustics.room.geometry import SYNTHESIS_BANDS

BAND_EDGES, BANDS = pyFDN.octave_bands()
N_BANDS = BANDS.size

# The estimator's per-band level matches the total energy in each band, so a white decay's level rises
# as sqrt(bandwidth): a spectral density needs these to become a level
BANDWIDTHS = BAND_EDGES[:, 1] - BAND_EDGES[:, 0]


def check_bands(bands: np.ndarray) -> None:
    """Raise if ``bands`` is not the package grid, which would misalign every band-indexed array."""
    bands = np.asarray(bands, dtype=float)
    if bands.shape != BANDS.shape or not np.allclose(bands, BANDS):
        raise ValueError(f"band grid must be {BANDS.tolist()} Hz, got {bands.tolist()}")


# RoomGeometry puts materials on SYNTHESIS_BANDS without importing pyFDN; the two must be one grid
check_bands(np.asarray(SYNTHESIS_BANDS))
