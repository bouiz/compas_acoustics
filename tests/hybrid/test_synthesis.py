import numpy as np
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.hybrid.bands import BANDS
from compas_acoustics.hybrid.early import encode
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.hybrid.synthesis import _select
from compas_acoustics.hybrid.synthesis import prepare
from compas_acoustics.hybrid.ambisonics import render_ambisonics
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.geometry import RoomGeometry

FS = 48000
MAX_ORDER = 2
TARGET = RoomTarget(rt=np.full(8, 0.6), level=np.ones(8), bands=BANDS.copy(), fs=float(FS))
WALL = AcousticMaterial("wall", [0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25], 0.1)


def setup(panel=False):
    scene = Scene("test")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
    if panel:
        mesh = Mesh.from_vertices_and_faces([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], [[0, 1, 2, 3]])
        scene.add_acoustic_object(AcousticObject(mesh, WALL, name="panel", role="obstacle"))
    geometry = RoomGeometry.from_scene(scene)
    source, receiver = Source("s", 2.0, 4.0, 1.5), Receiver("r", 8.0, 4.0, 1.5)
    room = geometry.to_pra(FS, MAX_ORDER)
    room.add_soundsource(source_to_pra(source))
    room.add_microphone_array(np.array([receiver.position], dtype=float).T)
    room.image_source_model()
    pair = Pair(geometry, source.position, receiver.position, receiver.rotation, float(FS), 343.0)
    return pair, image_sources(room, 0, 0, pair)


def test_the_response_is_early_plus_late():
    pair, iss = setup()
    result = render_ambisonics(prepare(pair, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5), normalization="N3D")
    assert result.rir.shape == (24000, 4)
    np.testing.assert_allclose(result.rir, result.early + result.late)
    np.testing.assert_allclose(result.early, encode(_select(iss, iss.order <= MAX_ORDER), 1, 24000))
    assert result.core.driving_images == int(np.sum(iss.order == MAX_ORDER))
    assert result.core.drive_gain == pytest.approx(np.sqrt(96))
    assert result.normalization == "N3D"
    assert np.sum(result.late**2) > 0.0


def test_sn3d_is_n3d_over_sqrt_2n_plus_1():
    pair, iss = setup()
    n3d = render_ambisonics(prepare(pair, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5), normalization="N3D")
    sn3d = render_ambisonics(prepare(pair, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5), normalization="SN3D")
    np.testing.assert_allclose(sn3d.rir, n3d.rir / np.sqrt([1.0, 3.0, 3.0, 3.0]))


def test_an_unknown_normalization_raises():
    pair, iss = setup()
    with pytest.raises(ValueError, match="normalization"):
        render_ambisonics(prepare(pair, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5), normalization="FuMa")


def test_diffraction_in_a_rectangular_room_without_obstacles_changes_nothing():
    pair, iss = setup()
    plain = render_ambisonics(prepare(pair, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5))
    bent = render_ambisonics(prepare(pair, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5, diffraction="udfa"))
    np.testing.assert_allclose(bent.early, plain.early)
    np.testing.assert_allclose(bent.late, plain.late)


def test_prepare_is_format_neutral():
    pair, iss = setup()
    core = prepare(pair, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5)
    assert core.lines.shape == (core.n_samples, 96) and core.late_envelope is None
    assert core.builds[0][0].C.shape == (1, 96) and core.line_directions.shape == (96, 3)
    first, second = render_ambisonics(core), render_ambisonics(core, sh_order=2, normalization="N3D")
    assert second.rir.shape == (core.n_samples, 9) and second.normalization == "N3D" and second.core is core
    np.testing.assert_allclose(second.early[:, 0], first.early[:, 0], rtol=0, atol=1e-12 * np.max(np.abs(first.early)))
    assert second.late_lines is core.lines  # no late envelope
    np.testing.assert_allclose(second.late[:, 0], core.lines.sum(axis=1) / np.sqrt(96), atol=1e-12)


def test_an_order_the_grid_cannot_carry_raises():
    pair, iss = setup()
    with pytest.raises(ValueError, match="leaks"):
        render_ambisonics(prepare(pair, iss, TARGET, max_order=MAX_ORDER, duration_s=0.2), sh_order=7)  # 96 lines carry orders up to 6


@pytest.mark.parametrize("render", ["ambisonics", "binaural"])
def test_each_diffracted_path_is_evaluated_once_per_rendering(monkeypatch, render):
    """UDFA's transfer functions are the expensive part: the core leaves them to the renderer, which needs W anyway."""
    import compas_acoustics.hybrid.diffraction as diffraction
    from compas_acoustics.hybrid.binaural import HrtfSet, render_binaural

    calls = []
    original = diffraction.encode_diffraction
    monkeypatch.setattr(diffraction, "encode_diffraction", lambda *a, **k: calls.append(1) or original(*a, **k))
    monkeypatch.setattr("compas_acoustics.hybrid.ambisonics.encode_diffraction", diffraction.encode_diffraction)
    monkeypatch.setattr("compas_acoustics.hybrid.binaural.encode_diffraction", diffraction.encode_diffraction)
    pair, iss = setup(panel=True)
    core = prepare(pair, iss, TARGET, max_order=MAX_ORDER, duration_s=0.2, diffraction="udfa")
    assert len(core.diffraction_paths) > 0 and calls == []
    hrirs = np.zeros((1, 2, 8))
    hrirs[:, :, 0] = 1.0
    if render == "ambisonics":
        render_ambisonics(core)
    else:
        render_binaural(core, HrtfSet(hrirs, np.array([[1.0, 0.0, 0.0]]), FS))
    assert calls == [1]
