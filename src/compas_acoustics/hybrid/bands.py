"""The octave band grid of the hybrid synthesis, shared with the decay estimators."""

from __future__ import annotations

import numpy as np

from compas_acoustics.room.geometry import SYNTHESIS_BANDS

try:
    import pyFDN

    _, BANDS = pyFDN.octave_bands()
except ImportError:  # the mono method's diffraction needs the grid, not the network
    BANDS = np.asarray(SYNTHESIS_BANDS, dtype=float)
N_BANDS = BANDS.size


def check_bands(bands: np.ndarray) -> None:
    """Raise if ``bands`` is not the package grid, which would misalign every band-indexed array."""
    bands = np.asarray(bands, dtype=float)
    if bands.shape != BANDS.shape or not np.allclose(bands, BANDS):
        raise ValueError(f"band grid must be {BANDS.tolist()} Hz, got {bands.tolist()}")


# RoomGeometry puts materials on SYNTHESIS_BANDS without importing pyFDN; the two must be one grid
check_bands(np.asarray(SYNTHESIS_BANDS))
