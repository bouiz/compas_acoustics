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


def test_the_materials_library_loads():
    paths = glob.glob(os.path.join(Defaults().MATERIALS_LIBRARY_PATH, "*.json"))
    assert paths
    for path in paths:
        AcousticMaterial.from_json(path)
