import pytest
from lab_shapes import sawtooth_tile

from compas_acoustics.lab import ScatteringSample
from compas_acoustics.lab import lab_material
from compas_acoustics.lab import measure_scattering
from compas_acoustics.materials import AcousticMaterial


@pytest.fixture(scope="module")
def scattering():
    pytest.importorskip("embreex")
    return measure_scattering(ScatteringSample.disc(sawtooth_tile(), diameter=0.4), method="geometric", preset="survey", band_range=(1000, 1000))


def test_material_from_numbers_and_a_result(scattering):
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.4]
    m = lab_material("test panel", alpha, scattering)
    assert m.absorption_coefficients == alpha
    assert len(m.scattering_coefficients) == 7
    assert "s by geometric (preset survey, heuristic)" in m.data_origin
    assert m.data_origin.startswith("compas_acoustics.lab: alpha_s given; s by geometric")


def test_one_absorption_value_fills_every_octave():
    m = lab_material("p", 0.3)
    assert m.absorption_coefficients == [0.3] * 7
    assert lab_material("p", [0.3]).absorption_coefficients == [0.3] * 7


def test_absorption_from_a_material():
    m = lab_material("p", AcousticMaterial("m", [0.3] * 7))
    assert m.absorption_coefficients == [0.3] * 7


def test_absorption_from_anything_with_absorption_coefficients():
    class Measured:
        absorption_coefficients = [0.2] * 7

    assert lab_material("p", Measured()).absorption_coefficients == [0.2] * 7


def test_the_name_category_and_description_are_kept():
    m = lab_material("p", 0.2, category="Wood", description="a panel")
    assert m.name == "p" and m.description == "a panel" and m.category.value == "Wood"


def test_without_scattering_the_material_has_none():
    m = lab_material("p", [0.2] * 7)
    assert not m.scattering_coefficients
    assert m.data_origin == "compas_acoustics.lab: alpha_s given; no scattering"


def test_out_of_range_values_are_clipped_with_a_warning():
    with pytest.warns(UserWarning, match=r"absorption coefficient 1\.08 in the 500 Hz octave clipped to 1") as record:
        m = lab_material("p", [0.2, 0.2, 1.08, 0.2, 0.2, -0.02, 0.2])
    assert m.absorption_coefficients[2] == 1.0 and m.absorption_coefficients[5] == 0.0
    assert any(r"absorption coefficient -0.02 in the 4000 Hz octave clipped to 0" in str(w.message) for w in record)


def test_a_nan_band_is_refused_naming_the_octave():
    with pytest.raises(ValueError, match="no absorption coefficient in the 2000 Hz octave"):
        lab_material("p", [0.2, 0.2, 0.2, 0.2, float("nan"), 0.2, 0.2])


def test_a_strongly_absorbing_sample_warns_about_the_rigid_scattering_assumption(scattering):
    with pytest.warns(UserWarning, match="ISO 17497-1 6.3.4: the sample absorbs more than 0.5, the scattering coefficient was computed for a rigid sample"):
        lab_material("p", [0.2, 0.2, 0.2, 0.7, 0.2, 0.2, 0.2], scattering)


def test_a_strongly_absorbing_sample_without_scattering_does_not_warn(recwarn):
    lab_material("p", [0.9] * 7)
    assert not [w for w in recwarn if "6.3.4" in str(w.message)]
