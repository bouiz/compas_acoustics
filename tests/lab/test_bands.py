import numpy as np
import pytest

from compas_acoustics.lab.bands import EXTENDED_BAND_RANGE
from compas_acoustics.lab.bands import band_average
from compas_acoustics.lab.bands import fill_octaves
from compas_acoustics.lab.bands import fold_to_octaves
from compas_acoustics.lab.bands import simulation_frequencies
from compas_acoustics.lab.bands import third_octave_centres


def test_default_is_the_iso_range_100_to_5000():
    nominal, exact = third_octave_centres()
    assert list(nominal) == [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
    assert np.allclose(exact, 1000 * 10 ** (np.arange(-10, 8) / 10))


def test_extended_range_reaches_8000():
    nominal, _ = third_octave_centres(EXTENDED_BAND_RANGE)
    assert list(nominal[-3:]) == [5000, 6300, 8000] and len(nominal) == 20


def test_range_outside_100_to_8000_raises():
    with pytest.raises(ValueError, match="100.*8000"):
        third_octave_centres((100, 10000))


def test_range_is_inclusive_and_a_single_band_is_valid():
    nominal, exact = third_octave_centres((1000, 1000))
    assert list(nominal) == [1000] and np.allclose(exact, [1000.0])
    nominal, _ = third_octave_centres((315, 630))
    assert list(nominal) == [315, 400, 500, 630]


def test_reversed_range_raises():
    with pytest.raises(ValueError, match="100.*8000"):
        third_octave_centres((1000, 500))


def test_three_frequencies_per_band_inside_the_band():
    f = simulation_frequencies()
    _, exact = third_octave_centres()
    assert f.shape == (54,)
    assert np.allclose(f[1::3], exact)
    assert np.allclose(f[0::3], exact * 2 ** (-1 / 9)) and np.allclose(f[2::3], exact * 2 ** (1 / 9))


def test_band_average_is_mean_of_triples():
    v = np.arange(6.0)
    assert np.allclose(band_average(v), [1.0, 4.0])


def test_band_average_works_on_the_last_axis():
    v = np.arange(12.0).reshape(2, 6)
    assert np.allclose(band_average(v), [[1.0, 4.0], [7.0, 10.0]])


def test_default_range_folds_to_complete_octaves_125_to_4000():
    nominal, _ = third_octave_centres()
    octaves, values = fold_to_octaves(np.arange(18.0), nominal)  # value i for band i
    assert octaves == [125, 250, 500, 1000, 2000, 4000]
    assert values[0] == pytest.approx((0 + 1 + 2) / 3)  # 125 octave: 100, 125, 160
    assert values[5] == pytest.approx((15 + 16 + 17) / 3)  # 4000 octave: 3150, 4000, 5000


def test_extended_range_adds_an_8000_octave_from_two_bands():
    nominal, _ = third_octave_centres(EXTENDED_BAND_RANGE)
    octaves, values = fold_to_octaves(np.arange(20.0), nominal)
    assert octaves[-1] == 8000 and values[-1] == pytest.approx((18 + 19) / 2)  # 6300, 8000 (10 kHz not simulated)


def test_fill_octaves_takes_the_nearest_simulated_octave():
    filled, pairs = fill_octaves([125, 250, 500, 1000, 2000, 4000], np.arange(6.0))
    assert np.allclose(filled, [0, 1, 2, 3, 4, 5, 5]) and pairs == [(8000, 4000)]
