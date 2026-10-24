import re
import sys
import time
import warnings

import numpy as np
import pytest
from compas import json_dumps
from compas import json_loads
from lab_shapes import bumped_box
from lab_shapes import flat_tile
from lab_shapes import flat_tile_with_zero_area_face
from lab_shapes import sawtooth_tile
from lab_shapes import sine_tile
from lab_shapes import v_groove_tile

from compas_acoustics.lab import scattering
from compas_acoustics.lab.bands import EXTENDED_BAND_RANGE
from compas_acoustics.lab.bands import THIRD_OCTAVE_CENTRES
from compas_acoustics.lab.grids import SourceGrid
from compas_acoustics.lab.results import ScatteringResult
from compas_acoustics.lab.sample import ScatteringSample
from compas_acoustics.lab.scattering import measure_scattering
from compas_acoustics.lab.scattering import preview_sources
from compas_acoustics.lab.solvers import KirchhoffSolver
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
    r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="kirchhoff", preset="survey", band_range=EXTENDED_BAND_RANGE)
    assert r.bands[-1] == 8000 and r.octave_bands[-1] == 8000 and r.filled_octaves == []
    assert any("beyond ISO" in w for w in r.warnings)


def test_default_preset_is_standard_and_an_explicit_one_overrides_it(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    sample = ScatteringSample.disc(flat_tile(), diameter=0.4)
    assert measure_scattering(sample, method="kirchhoff", band_range=(1000, 1000)).generation["preset"] == "standard"
    assert measure_scattering(sample, method="bem", band_range=(1000, 1000)).generation["preset"] == "standard"
    assert measure_scattering(sample, method="bem", preset="survey", band_range=(1000, 1000)).generation["preset"] == "survey"


def test_unknown_method_names_the_supported_ones():
    with pytest.raises(ValueError, match="kirchhoff.*bem"):
        measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="turntable")


def test_warnings_are_also_emitted(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="kirchhoff", preset="survey", band_range=(4000, 4000))
    assert "the 8000 Hz octave was not simulated; the material takes the 4000 Hz value" in r.warnings
    assert r.filled_octaves == [125, 250, 500, 1000, 2000, 8000]
    assert set(r.warnings) <= {str(w.message) for w in caught}


def test_non_finite_pressures_raise_naming_the_method(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: NanSolver())
    with pytest.raises(RuntimeError, match="kirchhoff"):
        measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="kirchhoff", preset="survey", band_range=(1000, 1000))


def test_kirchhoff_end_to_end_sine_beats_flat():
    flat = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="kirchhoff", preset="survey", band_range=(4000, 4000))
    sine = measure_scattering(ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4), method="kirchhoff", preset="survey", band_range=(4000, 4000))
    assert max(flat.s_bands) < 0.05 and sine.s_bands[-1] > 0.3


def test_result_has_no_nan_with_degenerate_faces():
    tile = flat_tile_with_zero_area_face()
    r = measure_scattering(ScatteringSample.disc(tile, diameter=0.4), method="kirchhoff", preset="survey", band_range=(500, 1000))
    assert np.isfinite(r.s_bands).all()


def test_result_round_trips_through_json():
    r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.3), method="kirchhoff", preset="survey", band_range=(500, 1000))
    t = json_loads(json_dumps(r))
    assert isinstance(t, ScatteringResult) and np.allclose(t.s_angle, r.s_angle) and t.generation["method"] == "kirchhoff"
    assert t.polar_source_indices == r.polar_source_indices and t.unconverged_octaves == r.unconverged_octaves == []


class UnconvergedSolver(FakeSolver):
    """Reports unconverged steps as BemSolver does: at 100·2^(-1/9) Hz (the 125 Hz octave) and at 630 Hz (the 500 Hz octave)."""

    def solve(self, sample, sources, receivers, frequencies, c):
        p = super().solve(sample, sources, receivers, frequencies, c)
        p.generation["unconverged"] = [
            {"body": "sample", "frequency": float(frequencies[0]), "iterations": 250, "residual": 0.08},
            {"body": "reference", "frequency": float(frequencies[0]), "iterations": 250, "residual": None},
            {"body": "sample", "frequency": float(frequencies[3 * 8 + 1]), "iterations": 250, "residual": 2.6e-5},
        ]
        return p


def test_unconverged_steps_reach_the_warnings_and_mark_their_octaves(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: UnconvergedSolver())
    with pytest.warns(UserWarning, match="did not converge"):
        r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="bem", preset="survey")
    assert r.unconverged_octaves == [125, 500]
    assert "bem: NumCalc did not converge for the sample and the reference at 93, 631 Hz; s in the 125, 500 Hz octaves is unreliable" in r.warnings
    t = json_loads(json_dumps(r))
    assert t.unconverged_octaves == [125, 500] and t.generation["unconverged"][1]["residual"] is None
    import json

    json.loads(json_dumps(r), parse_constant=lambda name: pytest.fail("not strict JSON: " + name))


def test_polar_levels_are_kept_for_the_preview_sources_only(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    sample = ScatteringSample.disc(flat_tile(), diameter=0.4)
    r = measure_scattering(sample, method="kirchhoff", preset="standard", band_range=(1000, 1000))
    assert r.s_angle.shape[0] == 97 and len(r.polar_source_indices) == 13 and r.polar_sample_db.shape == (13, 1, len(r.receiver_directions))
    survey = SourceGrid.preset("survey")
    used = np.column_stack([r.source_colatitude, r.source_azimuth])[r.polar_source_indices]
    assert sorted(map(tuple, used)) == sorted(zip(survey.colatitude, survey.azimuth))  # "standard" holds the survey directions
    rotational = ScatteringSample.disc(flat_tile(), diameter=0.4, symmetry="rotational")
    assert preview_sources(SourceGrid.preset("fine", "rotational"), "rotational") == [0, 11, 23]  # 1.26°, 28.98°, 59.22°: nearest 0°, 30°, 60°
    assert measure_scattering(rotational, method="kirchhoff", preset="standard", band_range=(1000, 1000)).polar_source_indices == [0, 3, 6]


def test_batched_kirchhoff_gives_the_unbatched_result(monkeypatch):
    sample = ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.3)
    whole = measure_scattering(sample, method="kirchhoff", preset="survey", band_range=(2000, 2000), workers=1)
    monkeypatch.setattr(scattering, "_BATCH_BYTES", 1)  # one source per batch
    calls = []
    original = KirchhoffSolver.solve_batches

    def spy(self, *args, **kwargs):
        calls.append(kwargs["batch_size"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(KirchhoffSolver, "solve_batches", spy)
    batched = measure_scattering(sample, method="kirchhoff", preset="survey", band_range=(2000, 2000), workers=1)
    assert calls == [1]
    assert np.array_equal(batched.s_angle, whole.s_angle) and batched.s_bands == whole.s_bands
    assert np.array_equal(batched.polar_sample_db, whole.polar_sample_db) and np.array_equal(batched.polar_reference_db, whole.polar_reference_db)


class SlowSolver(FakeSolver):
    def estimate_seconds(self, sample, sources, receivers, frequencies):
        return 7200.0


def test_a_long_run_is_warned_about_before_it_starts(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: SlowSolver())
    with pytest.warns(UserWarning, match=r"kirchhoff: this run takes about 2.0 h \(13 sources x 3 frequencies"):
        measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="kirchhoff", preset="survey", band_range=(1000, 1000))


def test_classes_are_exported_from_the_package():
    import compas_acoustics

    assert compas_acoustics.ScatteringResult is ScatteringResult and compas_acoustics.ScatteringSample is ScatteringSample


def test_the_correlation_methods_return_d(monkeypatch):
    with monkeypatch.context() as patched:
        patched.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
        same = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="kirchhoff", preset="survey")
    assert np.allclose(same.d_octaves, 0.0, atol=1e-9)  # sample == reference -> d_n = 0
    assert same.d_angle.shape == same.s_angle.shape and len(same.d_bands) == len(same.bands) and len(same.d_octaves) == len(same.octave_bands)
    sine = measure_scattering(ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4), method="kirchhoff", preset="survey", band_range=(2000, 4000))
    assert np.isfinite(sine.d_angle).all()
    # measured 2026-10-24: d_octaves 0.0093 (2 kHz), 0.0175 (4 kHz); d_n may be negative in general (a sample less uniform than its reference)
    assert all(0.0 <= d <= 1.0 for d in sine.d_octaves) and max(sine.d_octaves) > 0.0


def test_geometric_is_the_default_method():
    assert scattering.METHODS[0] == "geometric" and scattering.DEFAULT_PRESETS["geometric"] == "standard"
    assert scattering.measure_scattering.__defaults__[0] == "geometric"


def heuristic_warning(result):
    return next(w for w in result.warnings if w.startswith("geometric: s is frequency-modelled"))


def test_geometric_flat_disc_scatters_nothing():
    pytest.importorskip("embreex")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = measure_scattering(ScatteringSample.disc(flat_tile(), 0.4), method="geometric", n_rays=2000)
    assert np.allclose(r.s_octaves, 0.0, atol=1e-9) and np.allclose(r.d_octaves, 0.0, atol=1e-9)
    assert r.method == "geometric" and r.generation["heuristic"] is True and r.generation["preset"] == "standard"
    assert any("heuristic" in w for w in r.warnings) and set(r.warnings) <= {str(w.message) for w in caught}
    # the methods the warning tells the user to use exist ("use a, b or c for the physics"; a method only planned may be named elsewhere)
    advice = re.search(r"use (.+?) for the physics", next(w for w in r.warnings if "heuristic" in w)).group(1)
    recommended = re.split(r",\s*|\s+or\s+", advice)
    assert recommended and set(recommended) <= set(scattering.METHODS), recommended
    assert "bin_deg" in heuristic_warning(r) and "every frequency" in heuristic_warning(r)  # d and the polar levels
    assert not any("lost" in w for w in r.warnings) and r.generation["lost"] == 0.0
    assert r.generation["reference"] == "flat" and r.generation["n_rays"] == 2000 and r.generation["receivers"] == len(r.receiver_directions)
    # the specular cell holds everything: 0 dB there, -300 dB (empty) elsewhere, the same at every octave
    assert r.polar_sample_db.shape == (13, len(r.octave_bands), len(r.receiver_directions))
    assert np.allclose(r.polar_sample_db.max(axis=2), 0.0) and np.array_equal(r.polar_sample_db, r.polar_reference_db)
    assert (r.polar_sample_db == -300.0).sum(axis=2).min() == len(r.receiver_directions) - 1
    assert np.array_equal(r.polar_sample_db[:, :1].repeat(len(r.octave_bands), axis=1), r.polar_sample_db)


def test_geometric_sawtooth_rises_with_frequency():
    pytest.importorskip("embreex")
    r = measure_scattering(ScatteringSample.disc(sawtooth_tile(), 0.4), method="geometric", n_rays=5000)
    assert np.all(np.diff(r.s_octaves) >= 0) and r.s_octaves[0] < 0.01 and r.s_octaves[-1] > 0.3
    assert len(r.scattering_coefficients) == 7 and r.s_angle.shape == (97, 18) and r.d_angle.shape == (97, 18)
    assert np.allclose(r.d_angle, r.d_angle[:, :1])  # d is the same at every band
    assert r.generation["depth"] == pytest.approx(0.0105, abs=0.001)
    assert not any("small against the wavelength" in w for w in r.warnings)  # the depth curve is 0.61 at 5 kHz


def test_geometric_runs_in_seconds():
    pytest.importorskip("embreex")
    start = time.perf_counter()
    measure_scattering(ScatteringSample.disc(sawtooth_tile(), 0.4), method="geometric", preset="standard", n_rays=5000)
    assert time.perf_counter() - start < 20.0


def test_geometric_deep_cavity_warns_about_lost_rays():
    pytest.importorskip("embreex")
    with pytest.warns(UserWarning, match="lost"):
        r = measure_scattering(ScatteringSample.disc(v_groove_tile(), 0.4), method="geometric", preset="survey", n_rays=2000, max_bounces=1)
    assert any("lost" in w for w in r.warnings) and r.generation["lost"] > 0.01 and r.generation["max_bounces"] == 1


def test_geometric_with_a_proxy_marks_the_result(monkeypatch):
    pytest.importorskip("embreex")
    seen = {}
    response, heuristic = scattering.geometric_response, scattering.heuristic_s

    def spy_response(*args, **kwargs):
        seen["response"] = response(*args, **kwargs)
        return seen["response"]

    def spy_heuristic(s_max, *args):
        seen["s_max"] = np.array(s_max)
        return heuristic(s_max, *args)

    monkeypatch.setattr(scattering, "geometric_response", spy_response)
    monkeypatch.setattr(scattering, "heuristic_s", spy_heuristic)
    body = ScatteringSample.disc(sawtooth_tile(), 0.4).sample
    # max_bounces=1: the rays that need a second reflection (the riser, the baffle beyond the rim) are lost, by both bodies alike
    r = measure_scattering(ScatteringSample.as_drawn(body, reference=body), method="geometric", preset="survey", n_rays=2000, max_bounces=1)
    assert any("proxy" in w for w in r.warnings) and r.generation["reference"] == "proxy"
    lost = seen["response"].lost_sample
    assert lost.max() > 0.01 and np.array_equal(lost, seen["response"].lost_reference)  # not vacuous: rays are lost
    # a body against itself: what is lost is lost in both and cancels in s_max itself (the depth is 0, so s(f) is 0 whatever s_max is)
    assert np.abs(seen["s_max"]).max() < 1e-12
    assert np.allclose(r.s_octaves, 0.0, atol=1e-9)


def test_geometric_sawtooth_warns_about_no_lost_rays():
    pytest.importorskip("embreex")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = measure_scattering(ScatteringSample.disc(sawtooth_tile(), 0.4), method="geometric", preset="standard", n_rays=5000)
    # measured 2026-10-24 at 5000 rays: at most 0.0002 of a source's rays lost (0.011 before horizontal exits were binned)
    assert r.generation["lost"] < 0.002
    assert not any("lost" in w for w in r.warnings) and not any("lost" in str(w.message) for w in caught)


def test_geometric_without_trimesh_names_the_way_out(monkeypatch):
    sample = ScatteringSample.disc(flat_tile(), 0.4)
    monkeypatch.setitem(sys.modules, "trimesh", None)  # import trimesh raises ImportError
    with pytest.raises(ImportError, match='pass method="kirchhoff" to run without it') as raised:
        measure_scattering(sample, method="geometric", preset="survey", n_rays=10)
    assert "Python >= 3.11" in str(raised.value) and isinstance(raised.value.__cause__, ImportError)
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    assert measure_scattering(sample, method="kirchhoff", preset="survey", band_range=(1000, 1000)).method == "kirchhoff"


def test_geometric_rejects_an_unknown_option():
    with pytest.raises(TypeError, match="workers"):
        measure_scattering(ScatteringSample.disc(flat_tile(), 0.4), method="geometric", workers=2)


def test_geometric_mounted_box_scatters_at_high_frequency():
    pytest.importorskip("embreex")
    # a box of uniform height: before the depth took the mean height difference, h was 0 and s was 0 in every octave although s_max is 0.27-0.94
    r = measure_scattering(ScatteringSample.as_drawn(bumped_box(size=1.0, thickness=0.5, bump=0.0)), method="geometric", preset="survey", n_rays=5000)
    assert r.generation["depth"] == pytest.approx(0.49, abs=1e-9)
    assert r.s_octaves[-1] > 0.3 and not any("small against the wavelength" in w for w in r.warnings)


def test_geometric_warns_when_the_depth_curve_hides_the_scattering():
    pytest.importorskip("embreex")
    # the sawtooth (h = 0.0105 m, mean s_max about 1) up to 1 kHz only: the depth curve at the top band is 1 - exp(-(k h)^2) = 0.04
    with pytest.warns(UserWarning, match="small against the wavelength"):
        r = measure_scattering(ScatteringSample.disc(sawtooth_tile(), 0.4), method="geometric", preset="survey", band_range=(100, 1000), n_rays=2000)
    message = next(w for w in r.warnings if "small against the wavelength" in w)
    assert "1000 Hz" in message and "use kirchhoff or bem" in message


def test_the_reference_kind_is_recorded_for_every_method(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    proxy = ScatteringSample.as_drawn(bumped_box(), reference=bumped_box(bump=0.0))
    assert measure_scattering(proxy, method="kirchhoff", preset="survey", band_range=(1000, 1000)).generation["reference"] == "proxy"
    flat = ScatteringSample.disc(flat_tile(), 0.4)
    assert measure_scattering(flat, method="bem", preset="survey", band_range=(1000, 1000)).generation["reference"] == "flat"


def test_geometric_without_embree_names_embreex_and_kirchhoff(monkeypatch):
    trimesh_ray = pytest.importorskip("trimesh.ray")
    monkeypatch.setattr(trimesh_ray, "has_embree", False)
    with pytest.raises(ImportError, match="embreex") as raised:
        measure_scattering(ScatteringSample.disc(flat_tile(), 0.4), method="geometric", preset="survey", n_rays=10)
    assert 'pass method="kirchhoff" to run without it' in str(raised.value)


def test_geometric_warns_when_a_backing_thicker_than_the_baseplate_counts_as_relief():
    pytest.importorskip("embreex")
    backing = ScatteringSample.disc(sawtooth_tile(), 0.4, baseplate=0.1).sample  # the relief on a 0.1 m backing, closed
    with pytest.warns(UserWarning, match="backing") as caught:
        r = measure_scattering(ScatteringSample.as_drawn(backing), method="geometric", preset="survey", n_rays=2000)
    message = next(w for w in r.warnings if "backing" in w)
    assert "baseplate" in message and "0.01" in message and "flush" in message  # the default baseplate and the remedy are named
    assert message in [str(w.message) for w in caught]  # through warnings.warn as well as result.warnings


def test_geometric_does_not_warn_about_a_backing_when_there_is_none_to_warn_about():
    pytest.importorskip("embreex")
    backing = ScatteringSample.disc(sawtooth_tile(), 0.4, baseplate=0.1).sample
    cases = {
        "an open relief": ScatteringSample.disc(sawtooth_tile(), 0.4),
        "the backing's thickness passed as baseplate": ScatteringSample.as_drawn(backing, baseplate=0.1),
        "a box of uniform height": ScatteringSample.as_drawn(bumped_box(size=1.0, thickness=0.5, bump=0.0)),
        "a flat sample": ScatteringSample.disc(flat_tile(), 0.4),
    }
    for name, sample in cases.items():
        r = measure_scattering(sample, method="geometric", preset="survey", n_rays=2000)
        assert not any("backing" in w for w in r.warnings), name
