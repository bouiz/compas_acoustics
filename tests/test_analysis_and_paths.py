"""Decay curves, reliability and channel choice in the analyser; band gains of specular paths; diffracted paths for
display; pyroomacoustics' material names."""

import warnings

import numpy as np
import pytest
from compas.datastructures import Mesh

import compas_acoustics as ca
from compas_acoustics.materials import pyroomacoustics_material_names
from compas_acoustics.rays import diffracted_paths

FS = 48000


def _decay(rt, seconds, seed=0, second_rt=None, knee_db=-20.0):
    """Exponentially decaying noise of reverberation time ``rt``, with a direct impulse; optionally a second, slower slope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    envelope = 10 ** (-3 * t / rt)
    if second_rt is not None:
        envelope = np.maximum(envelope, 10 ** (knee_db / 20) * 10 ** (-3 * t / second_rt))
    ir = rng.standard_normal(t.size) * envelope * 0.05
    ir[0] = 1.0
    return ir


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def test_decay_curves_start_at_zero_and_fall():
    curves = ca.Analyser(_decay(0.8, 2.0)).decay_curves()
    assert curves["freqs"] == [125, 250, 500, 1000, 2000, 4000, 8000]
    for edc in curves["edc_db"]:
        assert edc[0] == pytest.approx(0.0)
        finite = edc[np.isfinite(edc)]
        assert np.all(np.diff(finite) <= 1e-9)
    assert curves["times"][1] == pytest.approx(1 / FS)


def test_reliability_flags_a_cut_and_a_bent_decay():
    full = ca.Analyser(_decay(0.8, 2.0)).reliability()
    assert all(full["T30_reliable"][2:6]) and all(full["EDT_reliable"])
    assert max(full["T30_xi"][2:6]) < 10.0
    cut = ca.Analyser(_decay(0.8, 0.3)).reliability()
    assert not any(cut["T30_reliable"])
    assert all(r < 45.0 for r in cut["range_db"])
    coupled = _quiet(ca.Analyser(_decay(1.0, 6.0, second_rt=4.0, knee_db=-20.0)).reliability)  # a coupled volume's knee
    assert min(coupled["T30_xi"][2:6]) > 10.0 and not any(coupled["T30_reliable"][2:6])


def test_the_analyser_reads_the_channel_asked_for():
    four = np.stack([_decay(0.8, 1.0, seed=k) for k in range(4)], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = ca.Analyser(four, channel=0)
        y = ca.Analyser(four, channel=1)
    np.testing.assert_array_equal(w.ir, four[:, 0])
    np.testing.assert_array_equal(y.ir, four[:, 1])
    with pytest.warns(UserWarning, match="channel 0 only"):
        ca.Analyser(four)
    with pytest.raises(ValueError):
        ca.Analyser(four, channel=4)
    with pytest.raises(ValueError):
        ca.Analyser(four[:, 0], channel=1)


def _room(with_panel=False):
    scene = ca.Scene("paths")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=ca.AcousticMaterial("wall", 0.36, 0.1)))
    if with_panel:
        panel = Mesh.from_vertices_and_faces([[4, 2, 0.5], [4, 4, 0.5], [4, 4, 2.5], [4, 2, 2.5]], [[0, 1, 2, 3]])
        scene.add_acoustic_object(ca.AcousticObject(panel, ca.AcousticMaterial("panel", 0.1, 0.1), name="panel", role="obstacle"))
    scene.add_source(ca.Source("s", 2, 3, 1.5))
    scene.add_receiver(ca.Receiver("r", 6, 3.2, 1.5))
    return scene


def test_specular_paths_carry_their_band_gain():
    reflections = _quiet(ca.SpecularReflections.from_scene, _room(), max_order=1)
    direct = next(p for p in reflections.paths if p.order == 0)
    length = float(np.linalg.norm(np.subtract([6, 3.2, 1.5], [2, 3, 1.5])))
    assert len(direct.band_gain) == 8
    np.testing.assert_allclose(direct.band_gain[:3], 1.0 / (4.0 * np.pi * length), rtol=1e-3)  # little air below 500 Hz
    assert direct.band_gain[-1] < direct.band_gain[0]  # air absorbs the high bands
    first = [p for p in reflections.paths if p.order == 1]
    for path in first:
        points = [np.asarray(list(p)) for p in path.points]
        travelled = sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:]))
        assert path.band_gain[2] == pytest.approx(np.sqrt(1 - 0.36) / (4 * np.pi * travelled), rel=3e-3)
    assert ca.SpecularPath.__from_data__(direct.__data__).band_gain == direct.band_gain


def test_diffracted_paths_bend_round_the_panel():
    scene = _room(with_panel=True)
    paths = _quiet(diffracted_paths, scene, max_order=0)
    assert paths
    for path in paths:
        assert len(path.points) == 3 and path.order == 0
        apex = np.asarray(path.points[1])
        on_edge = np.isclose(apex[0], 4.0) and (np.isclose(apex[1], 2.0) or np.isclose(apex[1], 4.0) or np.isclose(apex[2], 0.5) or np.isclose(apex[2], 2.5))
        assert on_edge
        assert path.distance == pytest.approx(sum(np.linalg.norm(np.subtract(b, a)) for a, b in zip(path.points[:-1], path.points[1:])), rel=1e-6)
        assert path.delay == pytest.approx(path.distance / ca.AirConditions().speed_of_sound)
    assert {p.zone for p in paths} == {"shadow"}  # the receiver stands behind the panel
    assert set(paths[0].to_dict()) >= {"points", "zone", "band_gain", "delay"}


def test_pyroomacoustics_materials_are_found_by_name(tmp_path):
    names = pyroomacoustics_material_names()
    assert "hard_surface" in names and len(names) > 50
    # a library without it falls back to pyroomacoustics' database
    material = ca.MaterialsLibrary(str(tmp_path)).material_from_name("hard_surface")
    assert material.data_origin == "pyroomacoustics" and len(material.absorption_coefficients) == 7
    assert material.absorption_coefficients == pytest.approx([0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05])
    # the package library reads its own file first
    assert "hard_surface" in ca.MaterialsLibrary().index
    six_bands = ca.AcousticMaterial.from_pyroomacoustics("lead_glazing")  # 125 Hz to 4 kHz in the database
    assert six_bands.absorption_coefficients[-1] == six_bands.absorption_coefficients[-2]
    with pytest.raises(ValueError):
        ca.AcousticMaterial.from_pyroomacoustics("no_such_material")
