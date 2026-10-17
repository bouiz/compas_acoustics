import numpy as np
import pyroomacoustics as pra
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.hybrid.bands import BANDS
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.hybrid.synthesis import synthesize
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.raytracing import ray_traced_rirs
from compas_acoustics.room_geometry import RoomGeometry

hybrid_rir = pytest.importorskip("hybrid_rir")

FS = 48000.0
SOURCE = np.array([2.0, 4.0, 1.5])
RECEIVER = np.array([8.0, 4.0, 1.5])
# The top two bands are equal, so upstream's linear extrapolation to 16 kHz and the hold rule coincide
WALL = AcousticMaterial("wall", [0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.25], [0.0] * 7)
PANEL = AcousticMaterial("panel", [0.05, 0.06, 0.08, 0.1, 0.12, 0.16, 0.16], [0.0] * 7)


def geometry(panel=False):
    scene = Scene("parity")
    scene.add_acoustic_object(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
    if panel:
        mesh = Mesh.from_vertices_and_faces([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], [[0, 1, 2, 3]])
        scene.add_acoustic_object(AcousticObject(mesh, PANEL, name="panel", role="obstacle"))
    return RoomGeometry.from_scene(scene)


def upstream_spec(g):
    """The same room as an upstream MeshGeometry: enclosure faces, then the drawn obstacles."""
    n_enclosure = len(g.faces) - 2 * len(g.obstacle_faces)
    faces = tuple(g.faces[:n_enclosure]) + tuple(g.obstacle_faces)
    names = tuple(g.face_materials[:n_enclosure]) + tuple(g.obstacle_face_materials)
    materials = {
        name: hybrid_rir.Material(g.materials[name].absorption, hybrid_rir.BANDS.copy(), g.materials[name].scattering)
        for name in dict.fromkeys(names)
    }
    mesh = hybrid_rir.MeshGeometry(
        vertices=g.vertices, faces=faces, face_materials=names, materials=materials,
        obstacles=tuple(dict.fromkeys(g.obstacle_face_materials)),
    )
    return hybrid_rir.RoomSpec(geometry=mesh, source=SOURCE, receiver=RECEIVER, fs=FS)


def test_ray_tracing_matches_upstream():
    g = geometry()
    ((ours,),) = ray_traced_rirs(
        g, [pra.SoundSource(SOURCE)], RECEIVER.reshape(3, 1), int(FS),
        max_order=3, duration=0.5, n_rays=2000, seed=1,
    )
    theirs = upstream_spec(g).ray_traced_rir(max_order=3, duration_s=0.5, n_rays=2000, seed=1)
    # 1.1e-9 of the peak measured at planning time (pyroomacoustics 0.10.1 in both)
    np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1e-7 * np.abs(theirs).max())


@pytest.mark.parametrize("panel", [False, True])
def test_synthesis_matches_upstream(panel):
    g = geometry(panel)
    room = g.to_pra(int(FS), 4)
    room.add_source(SOURCE.tolist())
    room.add_microphone(RECEIVER.reshape(3, 1))
    room.image_source_model()
    spec = Pair(g, SOURCE, RECEIVER, np.eye(3), FS)
    ours = synthesize(
        spec, image_sources(room, 0, 0, spec),
        RoomTarget(rt=np.full(8, 0.6), level=np.ones(8), bands=BANDS.copy(), fs=FS),
        max_order=3, duration_s=0.4, diffraction=panel,
    )
    theirs = hybrid_rir.synthesize(
        upstream_spec(g), max_order=3, duration_s=0.4, diffraction=panel,
        target=hybrid_rir.RoomTarget(rt=np.full(8, 0.6), level=np.ones(8), bands=hybrid_rir.BANDS.copy(), fs=FS),
    )
    np.testing.assert_allclose(ours.early, theirs.early, atol=1e-10)
    np.testing.assert_allclose(ours.rir, theirs.rir, atol=1e-8)


def test_btms_matches_upstream():
    g = geometry(panel=True)
    room = g.to_pra(int(FS), 1)
    room.add_source(SOURCE.tolist())
    room.add_microphone(RECEIVER.reshape(3, 1))
    room.image_source_model()
    spec = Pair(g, SOURCE, RECEIVER, np.eye(3), FS)
    target = RoomTarget(rt=np.full(8, 0.6), level=np.ones(8), bands=BANDS.copy(), fs=FS)
    ours = synthesize(spec, image_sources(room, 0, 0, spec), target, max_order=0, duration_s=0.3, diffraction="btms")
    theirs = hybrid_rir.synthesize(
        upstream_spec(g), max_order=0, duration_s=0.3, diffraction="btms",
        target=hybrid_rir.RoomTarget(rt=np.full(8, 0.6), level=np.ones(8), bands=hybrid_rir.BANDS.copy(), fs=FS),
    )
    np.testing.assert_allclose(ours.early, theirs.early, atol=1e-10)


def test_the_eyring_tail_matches_upstreams_default_target():
    """Without a target, upstream's synthesize uses RoomTarget.from_room: Hybrid(tail="eyring")."""
    from compas_acoustics import Receiver
    from compas_acoustics import Source
    from compas_acoustics.methods import Hybrid

    scene = Scene("parity")
    scene.add_acoustic_object(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
    scene.add_source(Source("s", *SOURCE))
    scene.add_receiver(Receiver("r", *RECEIVER))
    hybrid = Hybrid(tail="eyring", duration=0.4, normalize=False)
    scene.simulate(hybrid)
    (ours,) = hybrid.results.values()
    theirs = hybrid_rir.synthesize(upstream_spec(RoomGeometry.from_scene(scene)), max_order=3, duration_s=0.4)
    np.testing.assert_allclose(ours.target.rt, theirs.target.rt, rtol=1e-12)
    np.testing.assert_allclose(ours.target.level, theirs.target.level, rtol=1e-12)
    np.testing.assert_allclose(ours.early, theirs.early, atol=1e-10)
    np.testing.assert_allclose(ours.late, theirs.late, atol=1e-10)
