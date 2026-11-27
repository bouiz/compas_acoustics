"""The octave-band grid of the hybrid synthesis."""

import numpy as np
import pytest

from compas_acoustics.hybrid.bands import BANDS, check_bands


def test_lowest_band_is_62_point_5_not_63():
    assert BANDS[0] == 62.5
    assert BANDS.shape == (8,)
    assert BANDS[-1] == 8000.0


def test_check_bands_rejects_a_different_grid():
    with pytest.raises(ValueError, match="band grid"):
        check_bands(np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]))


def test_check_bands_rejects_wrong_length():
    with pytest.raises(ValueError, match="band grid"):
        check_bands(np.array([125.0, 250.0, 500.0]))
