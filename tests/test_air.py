"""The air: Cramer's speed of sound and ISO 9613-1 air absorption, through pyfar.constants."""

import sys

import numpy as np
import pytest

from compas_acoustics import AirConditions
from compas_acoustics.diffuse import _air_absorption_coefficient
from compas_acoustics.air import speed_of_sound

try:
    import pyfar.constants  # noqa: F401

    HAS_PYFAR_CONSTANTS = True
except ImportError:
    HAS_PYFAR_CONSTANTS = False

DB_PER_KM = 10.0 * np.log10(np.e) * 1000.0
# ISO 9613-1:1993 exact midband frequency of each nominal band, equation (6)
EXACT = {63: -12, 125: -9, 250: -6, 500: -3, 1000: 0, 2000: 3, 4000: 6, 8000: 9}
# ISO 9613-1:1993 Table 1, 101.325 kPa, dB/km
TABLE_1 = {
    (20.0, 10.0): {1000: 14.1, 2000: 45.3, 4000: 109.0, 8000: 175.0},
    (20.0, 50.0): {63: 0.123, 125: 0.445, 250: 1.32, 500: 2.73, 1000: 4.66, 2000: 9.86, 4000: 29.4, 8000: 104.0},
    (20.0, 90.0): {1000: 5.30, 2000: 9.06, 4000: 20.2, 8000: 62.6},
    (10.0, 70.0): {63: 0.122, 125: 0.411, 250: 1.04, 500: 1.93, 1000: 3.66, 2000: 9.66, 4000: 32.8, 8000: 117.0},
}


def exact(nominal):
    return np.array([1000.0 * 10.0 ** (0.1 * EXACT[f]) for f in nominal])


@pytest.mark.parametrize("conditions", list(TABLE_1))
def test_air_absorption_reproduces_iso_9613_1_table_1(conditions):
    t, h = conditions
    nominal = list(TABLE_1[conditions])
    alpha = AirConditions(temperature=t, humidity=h).air_attenuation(exact(nominal)) * DB_PER_KM
    np.testing.assert_allclose(alpha, [TABLE_1[conditions][f] for f in nominal], rtol=5e-3)


def test_without_a_medium_the_air_is_iso_9613_1_at_20_c_and_50_percent():
    from compas_acoustics.diffuse import air_attenuation
    from compas_acoustics.room_geometry import SYNTHESIS_BANDS

    default = AirConditions(temperature=20.0, humidity=50.0, pressure=101325.0)
    assert AirConditions() == default
    np.testing.assert_array_equal(air_attenuation(None), default.air_attenuation(SYNTHESIS_BANDS))
    np.testing.assert_array_equal(_air_absorption_coefficient(), default.air_attenuation(SYNTHESIS_BANDS))


# RAVEN's calculateSoundSpeed (Cramer 1993, 400 ppm CO2); pyfar takes 425 ppm, 0.003 m/s slower
@pytest.mark.parametrize(("t", "h", "p", "c"), [
    (20.0, 50.0, 101325.0, 343.987),
    (0.0, 0.0, 101325.0, 331.448),
    (20.0, 0.0, 101325.0, 343.359),
])
def test_speed_of_sound_is_cramers(t, h, p, c):
    assert AirConditions(temperature=t, humidity=h, pressure=p).speed_of_sound == pytest.approx(c, abs=5e-3)


def test_dry_air_follows_the_ideal_gas_law_over_cramers_range():
    c0 = AirConditions(temperature=0.0, humidity=0.0).speed_of_sound
    for t in (5.0, 15.0, 25.0, 30.0):
        expected = c0 * np.sqrt((273.15 + t) / 273.15)
        assert AirConditions(temperature=t, humidity=0.0).speed_of_sound == pytest.approx(expected, rel=5e-4)


def test_humid_air_is_faster_and_absorbs_differently():
    dry, humid = AirConditions(humidity=10.0), AirConditions(humidity=90.0)
    assert humid.speed_of_sound > dry.speed_of_sound
    assert humid.air_attenuation([8000.0])[0] < dry.air_attenuation([8000.0])[0]


@pytest.mark.parametrize("conditions", [dict(humidity=0.0), dict(temperature=0.0), dict(temperature=30.0),
                                        dict(pressure=75e3), dict(pressure=102e3), dict(humidity=100.0)])
def test_the_edges_of_the_valid_range_are_finite(conditions):
    air = AirConditions(**conditions)
    m = air.air_attenuation(np.geomspace(62.5, 16000.0, 9))
    assert np.all(np.isfinite(m)) and np.all(m > 0.0)
    assert np.isfinite(air.speed_of_sound) and air.speed_of_sound > 300.0


@pytest.mark.parametrize("conditions", [dict(temperature=-1.0), dict(temperature=31.0), dict(humidity=-1.0),
                                        dict(humidity=101.0), dict(pressure=74e3), dict(pressure=103e3)])
def test_outside_the_valid_range_raises(conditions):
    with pytest.raises(ValueError):
        AirConditions(**conditions)


def test_round_trip_through_a_dict():
    air = AirConditions(temperature=12.5, humidity=40.0, pressure=95000.0)
    assert AirConditions.from_dict(air.to_dict()) == air
    assert AirConditions.from_dict(None) is None


def test_without_a_medium_the_speed_of_sound_is_the_default_mediums():
    assert speed_of_sound(None) == AirConditions().speed_of_sound == pytest.approx(343.98, abs=0.01)


def test_predicted_rt_follows_the_medium():
    from compas_acoustics import AcousticMaterial, AcousticObjectsLibrary, Scene
    from compas_acoustics.diffuse import predicted_rt
    from compas_acoustics.room_geometry import RoomGeometry

    scene = Scene("rt")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(20, 15, 8, materials=AcousticMaterial("w", 0.05, 0.1)))
    geometry = RoomGeometry.from_scene(scene)
    default = predicted_rt(geometry, warn=False)
    np.testing.assert_array_equal(predicted_rt(geometry, warn=False, air=AirConditions()), default)
    dry = predicted_rt(geometry, warn=False, air=AirConditions(temperature=20.0, humidity=10.0))
    # Drier air absorbs more at 8 kHz than at 50 %: a shorter top band
    assert dry[-1] < default[-1]


def _scene():
    from compas_acoustics import AcousticMaterial, AcousticObjectsLibrary, Receiver, Scene, Source

    scene = Scene("air")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.3, 0.1)))
    scene.add_source(Source("src", 1.5, 1.5, 1.6))
    scene.add_receiver(Receiver("rec", 5.0, 3.0, 1.5))
    return scene


def test_the_direct_sound_arrives_at_the_mediums_speed():
    from compas_acoustics.simulation_methods import Pyroomacoustics

    air = AirConditions(temperature=0.0)
    (ir,) = _scene().simulate(Pyroomacoustics(air=air)).values()
    distance = float(np.linalg.norm(np.subtract(ir.receiver.position, ir.source.position)))
    expected = round(distance / air.speed_of_sound * ir.sample_rate)
    assert ir.direct_sound_index() == expected
    assert abs(int(np.argmax(np.abs(ir.denormalized()))) - expected) <= 1
    assert ir.generation["air"] == air.to_dict()
    assert ir.generation["speed_of_sound"] == pytest.approx(air.speed_of_sound)


def test_without_a_medium_the_generation_records_the_default():
    from compas_acoustics.simulation_methods import Pyroomacoustics

    (ir,) = _scene().simulate(Pyroomacoustics()).values()
    assert ir.generation["air"] == AirConditions().to_dict()
    assert ir.generation["speed_of_sound"] == pytest.approx(AirConditions().speed_of_sound)


def test_the_medium_survives_export(tmp_path):
    from compas_acoustics import ImpulseResponse
    from compas_acoustics.simulation_methods import Pyroomacoustics

    air = AirConditions(temperature=0.0)
    (ir,) = _scene().simulate(Pyroomacoustics(air=air)).values()
    wav_path, _ = ir.to_wav(str(tmp_path / "ir.wav"))
    loaded = ImpulseResponse.from_wav(wav_path)
    assert AirConditions.from_dict(loaded.generation["air"]) == air
    assert loaded.direct_sound_index() == ir.direct_sound_index()


def test_a_default_simulation_runs_without_pyfar_constants(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyfar.constants", None)
    air = AirConditions()
    assert 343.0 < air.speed_of_sound < 344.5
    assert np.all(np.diff(air.air_attenuation([125, 1000, 8000])) > 0)


@pytest.mark.skipif(not HAS_PYFAR_CONSTANTS, reason="pyfar < 0.8")
@pytest.mark.parametrize("t, h, p", [(0.0, 0.0, 75e3), (19.5, 41.7, 101325.0), (30.0, 100.0, 102e3)])
def test_air_matches_pyfar_constants(t, h, p):
    from pyfar import constants

    from compas_acoustics import _pyfar_constants as ours

    f = np.array([62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    np.testing.assert_allclose(ours.speed_of_sound_cramer(t, h / 100, atmospheric_pressure=p),
                               constants.speed_of_sound_cramer(t, h / 100, atmospheric_pressure=p), rtol=1e-12)
    np.testing.assert_allclose(ours.air_attenuation(t, f, h / 100, atmospheric_pressure=p)[1].freq,
                               constants.air_attenuation(t, f, h / 100, atmospheric_pressure=p)[1].freq, rtol=1e-12)
