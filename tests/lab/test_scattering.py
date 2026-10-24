import warnings

import numpy as np
import pytest
from compas import json_dumps
from compas import json_loads
from lab_shapes import flat_tile
from lab_shapes import flat_tile_with_zero_area_face
from lab_shapes import sine_tile

from compas_acoustics.lab import scattering
from compas_acoustics.lab.bands import EXTENDED_BAND_RANGE
from compas_acoustics.lab.bands import THIRD_OCTAVE_CENTRES
from compas_acoustics.lab.results import ScatteringResult
from compas_acoustics.lab.sample import ScatteringSample
from compas_acoustics.lab.scattering import measure_scattering
from compas_acoustics.lab.solvers import Pressures
from compas_acoustics.lab.solvers import ScatteringSolver


class FakeSolver(ScatteringSolver):
    name = "fake"

    def solve(self, sample, sources, receivers, frequencies, c):
        rng = np.random.default_rng(1)
        p0 = rng.normal(size=(sources.n, len(receivers.weights), len(frequencies))) + 0j
        return Pressures(p0.copy(), p0, frequencies, {"method": "fake"})

    def warnings(self, *args):
        return []


class NanSolver(FakeSolver):
    def solve(self, sample, sources, receivers, frequencies, c):
        p = super().solve(sample, sources, receivers, frequencies, c)
        p.sample[0, 0, 0] = np.nan
        return p


def test_shapes_and_bands(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="kirchhoff", preset="survey")
    assert r.bands == list(THIRD_OCTAVE_CENTRES[:18]) and len(r.s_bands) == 18  # default: ISO 100-5000
    assert r.octave_bands == [125, 250, 500, 1000, 2000, 4000] and len(r.scattering_coefficients) == 7
    assert r.filled_octaves == [8000] and r.scattering_coefficients[6] == r.scattering_coefficients[5]
    assert any("8000 Hz octave was not simulated" in w for w in r.warnings)
    assert r.s_angle.shape == (13, 18) and r.polar_sample_db.shape[:2] == (13, 6)
    assert np.allclose(r.s_octaves, 0.0, atol=1e-9)  # identical pressures -> no scattering
    assert np.isclose(r.polar_reference_db.max(axis=2), 0.0).all()


def test_extended_range_simulates_8000_and_warns_beyond_iso(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), preset="survey", band_range=EXTENDED_BAND_RANGE)
    assert r.bands[-1] == 8000 and r.octave_bands[-1] == 8000 and r.filled_octaves == []
    assert any("beyond ISO" in w for w in r.warnings)


def test_default_preset_depends_on_the_method(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    sample = ScatteringSample.disc(flat_tile(), diameter=0.4)
    assert measure_scattering(sample, method="kirchhoff", band_range=(1000, 1000)).generation["preset"] == "fine"
    assert measure_scattering(sample, method="bem", band_range=(1000, 1000)).generation["preset"] == "standard"
    assert measure_scattering(sample, method="bem", preset="survey", band_range=(1000, 1000)).generation["preset"] == "survey"


def test_unknown_method_names_the_supported_ones():
    with pytest.raises(ValueError, match="kirchhoff.*bem"):
        measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="turntable")


def test_warnings_are_also_emitted(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), preset="survey", band_range=(4000, 4000))
    assert "the 8000 Hz octave was not simulated; the material takes the 4000 Hz value" in r.warnings
    assert r.filled_octaves == [125, 250, 500, 1000, 2000, 8000]
    assert set(r.warnings) <= {str(w.message) for w in caught}


def test_non_finite_pressures_raise_naming_the_method(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: NanSolver())
    with pytest.raises(RuntimeError, match="kirchhoff"):
        measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), preset="survey", band_range=(1000, 1000))


def test_kirchhoff_end_to_end_sine_beats_flat():
    flat = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), preset="survey", band_range=(1000, 4000))
    sine = measure_scattering(ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4), preset="survey", band_range=(1000, 4000))
    assert max(flat.s_bands) < 0.05 and sine.s_bands[-1] > 0.3


def test_result_has_no_nan_with_degenerate_faces():
    tile = flat_tile_with_zero_area_face()
    r = measure_scattering(ScatteringSample.disc(tile, diameter=0.4), preset="survey", band_range=(500, 1000))
    assert np.isfinite(r.s_bands).all()


def test_result_round_trips_through_json():
    r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.3), preset="survey", band_range=(500, 1000))
    t = json_loads(json_dumps(r))
    assert isinstance(t, ScatteringResult) and np.allclose(t.s_angle, r.s_angle) and t.generation["method"] == "kirchhoff"


def test_classes_are_exported_from_the_package():
    import compas_acoustics

    assert compas_acoustics.ScatteringResult is ScatteringResult and compas_acoustics.ScatteringSample is ScatteringSample
