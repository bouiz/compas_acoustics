import glob
import os

import numpy as np
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics.defaults import Defaults


def test_zero_is_a_value_not_an_absence():
    m = AcousticMaterial("m", 0.0, 0.0)
    assert m.absorption_coefficients == [0.0] * 7
    assert m.scattering_coefficients == [0.0] * 7


def test_absent_scattering_is_none():
    assert AcousticMaterial("m", 0.1).scattering_coefficients is None
    assert AcousticMaterial("m", 0.1, []).scattering_coefficients is None


def test_absorption_is_required():
    with pytest.raises(ValueError):
        AcousticMaterial("m", None)


def test_material_bands_must_be_the_package_bands():
    material = AcousticMaterial("m", 0.1)
    material.frequency_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    with pytest.raises(ValueError, match="bands"):
        material.frequency_bands = [100, 200, 400, 800, 1600, 3200, 6400]


@pytest.mark.parametrize("values", [np.full(5, 0.2), np.full(7, 1.2), [0.2] * 5, [0.2] * 6 + [-0.1], 1.5, [0.2] * 6 + [float("nan")]])
def test_bad_coefficients_raise(values):
    with pytest.raises(ValueError):
        AcousticMaterial("m", values)


@pytest.mark.parametrize("kind", ["scattering", "transmission"])
def test_scattering_and_transmission_are_validated_too(kind):
    with pytest.raises(ValueError):
        AcousticMaterial("m", 0.2, **{f"{kind}_coefficients": [0.2] * 6 + [1.1]})


def test_numpy_coefficients_become_lists_of_float():
    m = AcousticMaterial("m", np.linspace(0.1, 0.7, 7), np.float64(0.3), np.full(7, 0.0))
    for values in (m.absorption_coefficients, m.scattering_coefficients, m.transmission_coefficients):
        assert type(values) is list and len(values) == 7 and all(type(v) is float for v in values)
    assert m.scattering_coefficients == [0.3] * 7


class _DotNetList:
    """Stands in for Grasshopper's System.Collections.Generic.List[Double]: iterable, not a Python list."""

    def __init__(self, values):
        self._values = list(values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


@pytest.mark.parametrize("kind", ["absorption", "scattering", "transmission"])
def test_any_iterable_is_a_list_of_coefficients(kind):
    kwargs = {"absorption_coefficients": 0.2, f"{kind}_coefficients": _DotNetList([0.5] * 7)}
    values = getattr(AcousticMaterial("m", **kwargs), f"{kind}_coefficients")
    assert type(values) is list and values == [0.5] * 7


@pytest.mark.parametrize("one", [[0.3], (0.3,), np.array([0.3]), _DotNetList([0.3])])
def test_one_value_applies_to_every_band(one):
    m = AcousticMaterial("m", one, one, one)
    assert m.absorption_coefficients == m.scattering_coefficients == m.transmission_coefficients == [0.3] * 7


def test_one_value_out_of_range_raises():
    with pytest.raises(ValueError):
        AcousticMaterial("m", _DotNetList([1.2]))


def test_an_empty_iterable_is_no_coefficients():
    assert AcousticMaterial("m", 0.1, _DotNetList([])).scattering_coefficients is None
    with pytest.raises(ValueError):
        AcousticMaterial("m", _DotNetList([]))


def test_the_materials_library_loads():
    paths = glob.glob(os.path.join(Defaults().MATERIALS_LIBRARY_PATH, "*.json"))
    assert paths
    for path in paths:
        AcousticMaterial.from_json(path)
