"""Source directivity: named patterns, measured ones (pyroomacoustics objects, SOFA files), per band, on every path."""

import os
import warnings

import numpy as np
import pyroomacoustics as pra
import pytest

import compas_acoustics as ca
from compas_acoustics.directivity import band_gains
from compas_acoustics.directivity import read_sofa
from compas_acoustics.directivity import to_pra
from compas_acoustics.room.pra import CARDIOID_P

FS = 48000


def _front_back(back_gain=0.1, n_points=300):
    """A measured pattern facing +x: a unit impulse in the front half, ``back_gain`` times one behind."""
    grid = pra.doa.GridSphere(n_points=n_points)
    irs = np.zeros((n_points, 1024))
    irs[:, 0] = np.where(grid.cartesian[0] >= 0.0, 1.0, back_gain)
    return pra.directivities.MeasuredDirectivity(pra.directivities.Rotation3D([0, 0], rot_order="yz"), grid, irs, FS)


def _scene(directivity, direction):
    scene = ca.Scene("directivity")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=ca.AcousticMaterial("wall", 0.3, 0.1)))
    scene.add_source(ca.Source("s", 2, 3, 1.5, directivity=directivity, direction=direction))
    scene.add_receiver(ca.Receiver("r", 6, 3, 1.5))
    return scene


def _direct_db(method, directivity):
    out = []
    for direction in ([1, 0, 0], [-1, 0, 0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ir = next(iter(_scene(directivity, direction).simulate(method).values())).denormalized()
        ir = ir if ir.ndim == 1 else ir[:, 0]
        n = int(4.0 / 343.0 * FS) + 40  # the direct sound, before the first reflection
        out.append(np.sum(ir[:n] ** 2))
    return 10.0 * np.log10(out[0] / out[1])


def test_named_patterns_stay_broadband():
    directions = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0]]).T
    assert to_pra("omnidirectional", [1, 0, 0], FS) is None
    for name, p in CARDIOID_P.items():
        if name == "omnidirectional":
            continue
        directivity = to_pra(name, [1, 0, 0], FS)
        gains = band_gains(directivity, directions)
        assert gains.shape == (3, 1)
        np.testing.assert_allclose(gains[:, 0], [1.0, p, 2 * p - 1.0], atol=1e-12)


def test_a_measured_pattern_gives_band_gains():
    measured = to_pra(_front_back(), [1, 0, 0], FS)
    gains = band_gains(measured, np.array([[1.0, 0, 0], [-1.0, 0, 0]]).T)
    assert gains.shape == (2, 8)
    np.testing.assert_allclose(20 * np.log10(gains), [[0.0] * 8, [-20.0] * 8], atol=1e-6)
    # a pyroomacoustics object is used as given, its own orientation kept whatever the source's direction
    np.testing.assert_allclose(band_gains(to_pra(_front_back(), [-1, 0, 0], FS), np.array([[1.0, 0, 0]]).T), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    "method",
    [
        ca.Mono(image_source_order=1, ray_tracing=False),
        ca.Mono(image_source_order=1, n_rays=3000, duration=0.5),
        ca.Ambisonics(sh_order=1, image_source_order=1, tail="eyring"),
    ],
    ids=["image sources", "ray tracing", "hybrid"],
)
def test_a_measured_source_is_heard_in_front_by_every_method(method):
    facing = _front_back(back_gain=0.1)
    away = pra.directivities.MeasuredDirectivity(pra.directivities.Rotation3D([0, 180], rot_order="yz"), facing._original_grid, facing._irs, FS)
    levels = []
    for directivity in (facing, away):  # the receiver is at +x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ir = next(iter(_scene(directivity, [1, 0, 0]).simulate(method).values())).denormalized()
        ir = ir if ir.ndim == 1 else ir[:, 0]
        levels.append(np.sum(ir[: int(4.0 / 343.0 * FS) + 40] ** 2))  # the direct sound, before the first reflection
    assert 10 * np.log10(levels[0] / levels[1]) == pytest.approx(20.0, abs=1.0)


def test_a_named_pattern_faces_the_source_direction():
    assert _direct_db(ca.Mono(image_source_order=1, ray_tracing=False), "cardioid") > 15.0


def test_diffracted_paths_take_the_source_pattern():
    from compas_acoustics.hybrid.diffraction import diffraction_paths
    from compas_acoustics.hybrid.pair import Pair

    scene = _scene("omnidirectional", [1, 0, 0])
    panel = ca.AcousticObject(_panel(), ca.AcousticMaterial("panel", 0.1, 0.1), name="panel", role="obstacle")
    scene.add_acoustic_object(panel)
    geometry = ca.RoomGeometry.from_scene(scene)
    spec = Pair(geometry=geometry, source=np.array([2.0, 3.0, 1.5]), receiver=np.array([6.0, 3.0, 1.5]), receiver_rotation=np.eye(3), fs=float(FS))
    plain = diffraction_paths(spec, max_order=0)
    halved = diffraction_paths(spec, max_order=0, source_gain=lambda direction: np.full(8, 0.5))
    assert plain and len(plain) == len(halved)
    for a, b in zip(plain, halved):
        np.testing.assert_allclose(b.band_gain, 0.5 * a.band_gain)


def _panel():
    from compas.datastructures import Mesh

    return Mesh.from_vertices_and_faces([[4, 2, 0.5], [4, 4, 0.5], [4, 4, 2.5], [4, 2, 2.5]], [[0, 1, 2, 3]])


def test_sofa_files_are_read_with_sofar():
    path = os.path.join(
        ca.Defaults().HRTFS_LIBRARY_PATH if hasattr(ca.Defaults(), "HRTFS_LIBRARY_PATH") else os.path.dirname(ca.__file__) + "/hrtfs_library", "KEMAR_FreeFieldComp_48kHz.sofa"
    )
    if not os.path.isfile(path):
        path = os.path.join(os.path.dirname(ca.__file__), "hrtfs_library", "KEMAR_FreeFieldComp_48kHz.sofa")
    ir, fs, sources, mics, _, _ = read_sofa(path, FS)
    assert fs == FS and ir.ndim == 3 and ir.shape[1] == 2  # (directions, ears, taps)
    assert sources.shape == (3, ir.shape[0]) and mics.shape[0] == 3
    with pytest.raises(ValueError, match="does not exist"):
        to_pra("missing.sofa", [1, 0, 0], FS)
    with pytest.raises(ValueError, match="not supported"):
        to_pra("kardioid", [1, 0, 0], FS)


def test_serialisation_and_the_worker_accept_names_and_files_only():
    source = ca.Source("s", 0, 0, 0, directivity="speaker.sofa#2")
    assert ca.Source.__from_data__(source.__data__).directivity == "speaker.sofa#2"
    with pytest.raises(ValueError, match="cannot be serialised"):
        ca.Source("s", 0, 0, 0, directivity=_front_back()).__data__  # noqa: B018
    from compas_acoustics.worker.api import make_source

    assert make_source("s", [0, 0, 0], directivity="speaker.sofa").directivity == "speaker.sofa"
    with pytest.raises(ValueError):
        make_source("s", [0, 0, 0], directivity="speaker.wav")


def test_reciprocity_keeps_a_directional_source_emitting():
    scene = _scene(_front_back(), [1, 0, 0])
    scene.add_source(ca.Source("t", 3, 2, 1.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ir = next(iter(scene.simulate(ca.Mono(image_source_order=1, ray_tracing=False, reciprocity=True)).values()))
    assert ir.generation["reciprocal"] is False
