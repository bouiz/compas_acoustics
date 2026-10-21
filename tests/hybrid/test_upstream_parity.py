import numpy as np
import pyroomacoustics as pra
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.hybrid.bands import BANDS
from compas_acoustics.hybrid.diffraction import btms_paths
from compas_acoustics.hybrid.diffraction import diffraction_edges
from compas_acoustics.hybrid.diffraction import diffraction_paths
from compas_acoustics.hybrid.diffraction import encode_btms
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.hybrid.synthesis import synthesize
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.simulation_methods.raytracing import ray_traced_rirs
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.geometry import SurfaceMaterial

hybrid_rir = pytest.importorskip("hybrid_rir")

FS = 48000.0
SOURCE = np.array([2.0, 4.0, 1.5])
RECEIVER = np.array([8.0, 4.0, 1.5])
# The top two bands are equal, so upstream's linear extrapolation to 16 kHz and the hold rule coincide
WALL = AcousticMaterial("wall", [0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.25], [0.0] * 7)
PANEL = AcousticMaterial("panel", [0.05, 0.06, 0.08, 0.1, 0.12, 0.16, 0.16], [0.0] * 7)


def geometry(panel=False):
    scene = Scene("parity")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
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
    from compas_acoustics.simulation_methods import Hybrid

    scene = Scene("parity")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
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


H = 3.0
# The transfer grid of upstream's tests/test_shell_edges.py
F = np.logspace(np.log10(20.0), np.log10(16000.0), 60)


def surface_materials(*materials):
    return {m.name: SurfaceMaterial.from_acoustic_material(m) for m in materials}


def upstream_materials(materials):
    return {name: hybrid_rir.Material(m.absorption, hybrid_rir.BANDS.copy(), m.scattering) for name, m in materials.items()}


def assert_paths_equal(ours, theirs, our_edges, their_edges):
    """Both sorted by distance; edges matched by position, since the edge lists may be ordered differently."""
    ours = sorted(ours, key=lambda p: (round(p.distance, 9), p.order, p.zone))
    theirs = sorted(theirs, key=lambda p: (round(p.distance, 9), p.order, p.zone))
    assert len(ours) == len(theirs) > 0
    for a, b in zip(ours, theirs):
        assert (a.order, a.zone) == (b.order, b.zone)
        np.testing.assert_array_equal(our_edges[a.edge].start, their_edges[b.edge].start)
        np.testing.assert_array_equal(our_edges[a.edge].end, their_edges[b.edge].end)
        np.testing.assert_allclose([a.delay, a.distance], [b.delay, b.distance], rtol=1e-12)
        np.testing.assert_allclose(a.doa, b.doa, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(a.band_gain, b.band_gain, rtol=1e-12)
        np.testing.assert_allclose(a.transfer(F), b.transfer(F), rtol=1e-12)


@pytest.mark.parametrize("receiver", [[4.0, 4.5, 1.7], [2.0, 4.5, 2.0]], ids=["shadow", "view"])
def test_room_corner_diffraction_matches_upstream(receiver):
    """The L room of upstream's test_shell_edges.py from the same arrays: same faces, so the same face indices."""
    outline = [(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)]
    n = len(outline)
    v = np.array([[x, y, 0.0] for x, y in outline] + [[x, y, H] for x, y in outline])
    faces = (tuple(range(n))[::-1], tuple(range(n, 2 * n))) + tuple(
        (k, (k + 1) % n, (k + 1) % n + n, k + n) for k in range(n))
    # Floor and ceiling of another material, so a reflection's gain depends on its face index
    names = ("panel", "panel") + ("wall",) * n
    materials = surface_materials(WALL, PANEL)
    g = RoomGeometry.from_mesh(v, faces, names, materials)
    mesh = hybrid_rir.MeshGeometry(vertices=v, faces=faces, face_materials=names,
                                   materials=upstream_materials(materials))
    source, receiver = np.array([3.0, 1.0, 1.5]), np.array(receiver)

    ours, theirs = diffraction_edges(g), hybrid_rir.diffraction_edges(mesh)
    assert len(ours) == len(theirs) == 1
    for a, b in zip(ours, theirs):
        for field in ("start", "end", "face_a", "face_b"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        assert a.theta_w == b.theta_w
        assert (a.faces, a.regions) == (b.faces, b.regions)

    spec = Pair(g, source, receiver, np.eye(3), FS)
    upstream = hybrid_rir.RoomSpec(geometry=mesh, source=source, receiver=receiver, fs=FS)
    our_paths, their_paths = diffraction_paths(spec, max_order=1), hybrid_rir.diffraction_paths(upstream, max_order=1)
    assert [(p.edge, p.order) for p in our_paths] == [(p.edge, p.order) for p in their_paths]
    assert_paths_equal(our_paths, their_paths, ours, theirs)

    w_ours = encode_btms(btms_paths(spec, max_order=1), 0, 4800, FS)
    w_theirs = hybrid_rir.encode_btms(hybrid_rir.btms_paths(upstream, max_order=1), 0, 4800, FS)
    assert np.max(np.abs(w_theirs)) > 0.0
    np.testing.assert_allclose(w_ours, w_theirs, rtol=0.0, atol=1e-12 * np.max(np.abs(w_theirs)))


def grid_room(xs, ys, solid, names):
    """Upstream's grid_room (tests/test_shell_edges.py), with the walls facing a solid cell named ``names[1]``."""
    index = {}

    def vid(x, y, z):
        return index.setdefault((x, y, z), len(index))

    nx, ny = len(xs) - 1, len(ys) - 1

    def air(i, j):
        return 0 <= i < nx and 0 <= j < ny and (i, j) not in solid

    def wall(i, j):
        return names[1] if (i, j) in solid else names[0]

    faces, face_names = [], []
    for i in range(nx):
        for j in range(ny):
            if not air(i, j):
                continue
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            faces += [(vid(x0, y0, 0), vid(x0, y1, 0), vid(x1, y1, 0), vid(x1, y0, 0)),
                      (vid(x1, y0, H), vid(x1, y1, H), vid(x0, y1, H), vid(x0, y0, H))]
            face_names += [names[0], names[0]]
            if not air(i + 1, j):
                faces.append((vid(x1, y0, 0), vid(x1, y1, 0), vid(x1, y1, H), vid(x1, y0, H)))
                face_names.append(wall(i + 1, j))
            if not air(i - 1, j):
                faces.append((vid(x0, y1, 0), vid(x0, y0, 0), vid(x0, y0, H), vid(x0, y1, H)))
                face_names.append(wall(i - 1, j))
            if not air(i, j + 1):
                faces.append((vid(x1, y1, 0), vid(x0, y1, 0), vid(x0, y1, H), vid(x1, y1, H)))
                face_names.append(wall(i, j + 1))
            if not air(i, j - 1):
                faces.append((vid(x0, y0, 0), vid(x1, y0, 0), vid(x1, y0, H), vid(x0, y0, H)))
                face_names.append(wall(i, j - 1))
    return np.array(sorted(index, key=index.get), float), tuple(faces), tuple(face_names)


def test_pillar_corners_diffract_as_upstream():
    """A pillar drawn as an outward-wound box: compas must wind it into itself for its corners to open to 270 deg.

    Upstream's room is tiled and wound by hand, so neither the winding nor the face order is taken from compas.
    """
    scene = Scene("pillar")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(10, 8, H, materials=WALL))
    pillar = Mesh.from_shape(Box.from_corner_corner_height([4, 3, 0], [5, 4, 0], H))
    scene.add_acoustic_object(AcousticObject(pillar, PANEL, name="pillar"))
    g = RoomGeometry.from_scene(scene)
    assert g.is_closed
    v, faces, names = grid_room([0, 4, 5, 10], [0, 3, 4, 8], {(1, 1)}, ("wall", "panel"))
    mesh = hybrid_rir.MeshGeometry(vertices=v, faces=faces, face_materials=names,
                                   materials=upstream_materials(g.materials))

    ours, theirs = diffraction_edges(g), hybrid_rir.diffraction_edges(mesh)
    assert len(ours) == len(theirs) == 4
    by_corner = {tuple(e.start): e for e in theirs}
    for a in ours:
        b = by_corner[tuple(a.start)]
        assert a.theta_w == b.theta_w == 1.5 * np.pi
        for field in ("end", "face_a", "face_b"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))

    source, receiver = np.array([2.0, 3.5, 1.5]), np.array([8.0, 3.5, 1.7])
    assert_paths_equal(
        diffraction_paths(Pair(g, source, receiver, np.eye(3), FS), max_order=1),
        hybrid_rir.diffraction_paths(hybrid_rir.RoomSpec(geometry=mesh, source=source, receiver=receiver, fs=FS), max_order=1),
        ours, theirs,
    )
