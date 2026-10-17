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
from compas_acoustics.hybrid.synthesis import synthesize
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.pra_converter import receivers_to_pra_mic_array
from compas_acoustics.pra_converter import source_to_pra
from compas_acoustics.room_geometry import RoomGeometry

FS = 48000
MAX_ORDER = 2
TARGET = RoomTarget(rt=np.full(8, 0.6), level=np.ones(8), bands=BANDS.copy(), fs=float(FS))
WALL = AcousticMaterial("wall", [0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25], 0.1)


def setup(panel=False):
    scene = Scene("test")
    scene.add_acoustic_object(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
    if panel:
        mesh = Mesh.from_vertices_and_faces([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], [[0, 1, 2, 3]])
        scene.add_acoustic_object(AcousticObject(mesh, WALL, name="panel", role="obstacle"))
    geometry = RoomGeometry.from_scene(scene)
    source, receiver = Source("s", 2.0, 4.0, 1.5), Receiver("r", 8.0, 4.0, 1.5)
    room = geometry.to_pra(FS, MAX_ORDER + 1)
    room.add_soundsource(source_to_pra(source))
    room.add_microphone_array(receivers_to_pra_mic_array(receiver, sample_rate=FS))
    room.image_source_model()
    spec = Pair(geometry, source.position, receiver.position, receiver.rotation, float(FS))
    return spec, image_sources(room, 0, 0, spec)


def test_the_response_is_early_plus_late():
    spec, iss = setup()
    result = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5, normalization="N3D")
    assert result.rir.shape == (24000, 4)
    np.testing.assert_allclose(result.rir, result.early + result.late)
    np.testing.assert_allclose(result.early, encode(_select(iss, iss.order <= MAX_ORDER), 1, 24000))
    assert result.driving_images == int(np.sum(iss.order == MAX_ORDER + 1))
    assert result.drive_gain == pytest.approx(4.0)
    assert result.normalization == "N3D"
    assert np.sum(result.late ** 2) > 0.0


def test_sn3d_is_n3d_over_sqrt_2n_plus_1():
    spec, iss = setup()
    n3d = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5, normalization="N3D")
    sn3d = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5, normalization="SN3D")
    np.testing.assert_allclose(sn3d.rir, n3d.rir / np.sqrt([1.0, 3.0, 3.0, 3.0]))


def test_an_unknown_normalization_raises():
    spec, iss = setup()
    with pytest.raises(ValueError, match="normalization"):
        synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5, normalization="FuMa")


def test_the_duration_defaults_to_one_and_a_half_reverberation_times():
    spec, iss = setup()
    assert synthesize(spec, iss, TARGET, max_order=MAX_ORDER).rir.shape[0] == int(np.ceil(0.9 * FS))


def test_a_non_positive_duration_raises():
    spec, iss = setup()
    with pytest.raises(ValueError, match="duration"):
        synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.0)


def test_without_a_reference_nothing_changes():
    spec, iss = setup()
    default = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5)
    none = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5, late_envelope=None)
    np.testing.assert_array_equal(default.rir, none.rir)


def test_a_reference_shapes_only_the_late_field():
    spec, iss = setup()
    plain = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5)
    reference = plain.early[:, 0] + 0.5 * plain.late[:, 0]
    shaped = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5, late_envelope=reference)
    np.testing.assert_array_equal(shaped.early, plain.early)
    assert np.sum(shaped.late ** 2) < np.sum(plain.late ** 2)


def test_diffraction_changes_only_the_early_field():
    spec, iss = setup(panel=True)
    plain = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5)
    bent = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5, diffraction=True)
    np.testing.assert_allclose(bent.late, plain.late)
    assert not np.allclose(bent.early, plain.early)


def test_diffraction_in_a_rectangular_room_without_obstacles_changes_nothing():
    spec, iss = setup()
    plain = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5)
    bent = synthesize(spec, iss, TARGET, max_order=MAX_ORDER, duration_s=0.5, diffraction=True)
    np.testing.assert_allclose(bent.early, plain.early)
    np.testing.assert_allclose(bent.late, plain.late)


def test_btms_changes_only_the_early_field():
    spec, iss = setup(panel=True)
    plain = synthesize(spec, iss, TARGET, max_order=0, duration_s=0.2)
    bent = synthesize(spec, iss, TARGET, max_order=0, duration_s=0.2, diffraction="btms")
    np.testing.assert_allclose(bent.late, plain.late)
    assert not np.allclose(bent.early, plain.early)


def test_an_unknown_diffraction_model_raises():
    spec, iss = setup(panel=True)
    with pytest.raises(ValueError, match="diffraction"):
        synthesize(spec, iss, TARGET, max_order=0, duration_s=0.1, diffraction="btm")
