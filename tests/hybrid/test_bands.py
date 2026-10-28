"""Ported from hybrid-rir (gpapadimitri/hybrid-rir @ a272da7, tests/test_bands.py)."""

import numpy as np
import pyFDN
import pytest

from compas_acoustics.hybrid.bands import BAND_EDGES, BANDS, check_bands


def test_bands_match_pyfdn_defaults():
    edges, centres = pyFDN.octave_bands()
    np.testing.assert_allclose(BANDS, centres)
    np.testing.assert_allclose(BAND_EDGES, edges)


def test_lowest_band_is_62_point_5_not_63():
    assert BANDS[0] == 62.5
    assert BANDS.shape == (8,)
    assert BANDS[-1] == 8000.0


def test_check_bands_accepts_the_package_grid():
    check_bands(BANDS.copy())


def test_check_bands_rejects_a_different_grid():
    with pytest.raises(ValueError, match="band grid"):
        check_bands(
            np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
        )


def test_check_bands_rejects_wrong_length():
    with pytest.raises(ValueError, match="band grid"):
        check_bands(np.array([125.0, 250.0, 500.0]))
