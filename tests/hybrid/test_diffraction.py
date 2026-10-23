"""Obstacle edges, diffraction paths and their visibility, rendering, and

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ a272da7, tests/test_diffraction.py).
diffraction in prepare and render_ambisonics."""

import warnings

import numpy as np
import pyfar as pf
import pytest

from compas_acoustics.hybrid.early import ImageSourceSet, encode
from _compat import RoomSpec, image_sources
from compas_acoustics.hybrid.synthesis import prepare
from compas_acoustics.hybrid.ambisonics import render_ambisonics
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.hybrid.btms import element_grid, wedge1st_ir, wedge2nd_ir
from compas_acoustics.hybrid.diffraction import (
    DiffractionPath,
    Edge,
    ObstacleEdgeWarning,
    Occluders,
    along_plane,
    btms_paths,
    diffraction_paths,
    edge_segments,
    encode_btms,
    encode_diffraction,
    obstacle_edges,
    pair_valid,
    second_order_start,
    segment_distance,
    toolbox_cyl,
    toolbox_frame,
    wedge_path,
)
from compas_acoustics.hybrid.bands import BANDS
from _compat import Material, MeshGeometry
from compas_acoustics.hybrid.udfa import udfa_transfer

C = 343.0
F = np.logspace(np.log10(20.0), np.log10(16000.0), 60)
PANEL = np.array([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], dtype=float)
TOP = 2.5


def corner():
    """Edge along z at the origin; the solid fills the +x, +y quadrant."""
    return Edge(
        start=np.array([0.0, 0.0, -1.0]), end=np.array([0.0, 0.0, 1.0]),
        face_a=np.array([1.0, 0.0, 0.0]), face_b=np.array([0.0, 1.0, 0.0]),
        theta_w=1.5 * np.pi,
    )


def at(theta, r, z=0.0):
    """Exterior angle theta from face_a, rotating away from the solid (via -y)."""
    return np.array([r * np.cos(theta), -r * np.sin(theta), z])


def test_cylindrical_coordinates_of_a_box_corner():
    path = wedge_path(corner(), at(np.pi / 4, 2.0), at(5 * np.pi / 4, 3.0))
    assert path.theta_s == pytest.approx(np.pi / 4)
    assert path.theta_r == pytest.approx(5 * np.pi / 4)
    assert (path.r_s, path.r_r, path.z_s, path.z_r, path.length) == pytest.approx((2, 3, 0, 0, 2))


def test_a_point_inside_the_solid_has_no_path():
    assert wedge_path(corner(), at(np.pi / 4, 2.0), np.array([1.0, 1.0, 0.0])) is None


def test_angles_are_measured_from_the_face_nearer_the_source():
    path = wedge_path(corner(), at(1.4 * np.pi, 2.0), at(0.3 * np.pi, 3.0))
    assert path.theta_s == pytest.approx(0.1 * np.pi)
    assert path.theta_r == pytest.approx(1.2 * np.pi)


def test_reciprocity():
    s, r = at(0.3, 2.0, -0.4), at(4.2, 3.0, 0.7)
    a = udfa_transfer(wedge_path(corner(), s, r), F, c=C)
    b = udfa_transfer(wedge_path(corner(), r, s), F, c=C)
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_face_order_does_not_matter():
    e = corner()
    flipped = Edge(e.start, e.end, e.face_b, e.face_a, e.theta_w)
    s, r = at(0.3, 2.0, -0.4), at(4.2, 3.0, 0.7)
    np.testing.assert_allclose(
        udfa_transfer(wedge_path(e, s, r), F, c=C),
        udfa_transfer(wedge_path(flipped, s, r), F, c=C), rtol=1e-10,
    )


def test_rigid_motion_invariance():
    rot = np.linalg.qr(np.random.default_rng(3).standard_normal((3, 3)))[0]
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    shift = np.array([3.0, -1.0, 2.0])
    e = corner()
    moved = Edge(rot @ e.start + shift, rot @ e.end + shift, rot @ e.face_a, rot @ e.face_b, e.theta_w)
    s, r = at(0.3, 2.0, -0.4), at(4.2, 3.0, 0.7)
    np.testing.assert_allclose(
        udfa_transfer(wedge_path(e, s, r), F, c=C),
        udfa_transfer(wedge_path(moved, rot @ s + shift, rot @ r + shift), F, c=C), rtol=1e-9,
    )


ROOM_V = np.array([[0, 0, 0], [10, 0, 0], [10, 8, 0], [0, 8, 0],
                   [0, 0, 4], [10, 0, 4], [10, 8, 4], [0, 8, 4]], dtype=float)
ROOM_F = [(0, 3, 7, 4), (1, 5, 6, 2), (0, 4, 5, 1), (3, 2, 6, 7), (0, 1, 2, 3), (4, 7, 6, 5)]
TILT = np.linspace(0.8, 1.2, 8)
SHELF_LO, SHELF_HI = np.array([4.0, 2.0, 0.0002]), np.array([4.6, 6.0, 1.523])


def box(lo, hi):
    v = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f  # bottom first


def room_with(obstacle_v, obstacle_f, name="shelf"):
    faces = ROOM_F + [tuple(8 + i for i in f) for f in obstacle_f]
    names = ["wall"] * 4 + ["floor", "ceiling"] + [name] * len(obstacle_f)
    # frequency-dependent on purpose: with every material flat, pyroomacoustics
    # returns a single band and image_sources refuses the mesh
    materials = {n: Material(a * TILT, BANDS.copy()) for n, a in
                 (("wall", 0.1), ("floor", 0.05), ("ceiling", 0.3), (name, 0.1))}
    return MeshGeometry(vertices=np.vstack([ROOM_V, obstacle_v]), faces=tuple(faces),
                        face_materials=tuple(names), materials=materials, obstacles=(name,))


def shelf_room(bottom=True, triangulate=False):
    v, f = box(SHELF_LO, SHELF_HI)
    if not bottom:
        f = f[1:]
    if triangulate:
        f = [t for q in f for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    return room_with(v, f)


def test_mesh_geometry_keeps_the_single_sided_obstacle_polygons():
    g = shelf_room()
    assert len(g.obstacle_faces) == 6
    assert set(g.obstacle_face_materials) == {"shelf"}
    np.testing.assert_allclose(g.vertices[list(g.obstacle_faces[1])][:, 2], 1.523)


@pytest.mark.parametrize("bottom, triangulate", [(True, False), (False, False), (True, True)])
def test_a_shelf_on_the_floor_has_four_top_and_four_vertical_edges(bottom, triangulate):
    edges = obstacle_edges(shelf_room(bottom, triangulate))
    assert len(edges) == 8
    assert all(e.theta_w == pytest.approx(1.5 * np.pi) for e in edges)
    tops = [e for e in edges if np.allclose([e.start[2], e.end[2]], 1.523)]
    assert len(tops) == 4
    assert sorted(round(e.length, 6) for e in tops) == [0.6, 0.6, 4.0, 4.0]


def test_a_panel_has_knife_edges():
    v = np.array([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], dtype=float)
    edges = obstacle_edges(room_with(v, [(0, 1, 2, 3)], "panel"))
    assert len(edges) == 4
    assert all(e.theta_w == pytest.approx(2 * np.pi) for e in edges)
    assert all(np.allclose(e.face_a, e.face_b) for e in edges)


def test_a_non_convex_component_is_skipped_with_a_warning():
    a_v, a_f = box(np.array([2.0, 2.0, 0.5]), np.array([3.0, 3.0, 1.0]))
    b_v, b_f = box(np.array([3.0, 2.0, 0.5]), np.array([3.5, 2.5, 1.5]))
    # two boxes sharing part of a face: one closed, non-convex component
    v = np.vstack([a_v, b_v])
    f = a_f + [tuple(8 + i for i in q) for q in b_f]
    with pytest.warns(ObstacleEdgeWarning):
        edges = obstacle_edges(room_with(v, f, "chair"))
    assert edges == []


def test_skipped_components_are_one_warning_that_counts_them():
    def two_boxes(shift):
        a_v, a_f = box(np.array([2.0, 2.0, 0.5]) + shift, np.array([3.0, 3.0, 1.0]) + shift)
        b_v, b_f = box(np.array([3.0, 2.0, 0.5]) + shift, np.array([3.5, 2.5, 1.5]) + shift)
        return np.vstack([a_v, b_v]), a_f + [tuple(8 + i for i in q) for q in b_f]

    (v1, f1), (v2, f2) = two_boxes(np.zeros(3)), two_boxes(np.array([0.0, 3.0, 0.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edges = obstacle_edges(room_with(np.vstack([v1, v2]), f1 + [tuple(16 + i for i in q) for q in f2], "chair"))
    assert edges == []
    skipped = [w for w in caught if issubclass(w.category, ObstacleEdgeWarning)]
    assert len(skipped) == 1
    assert "2 obstacle components (chair: 2)" in str(skipped[0].message)


def test_occluders_block_through_the_shelf_and_not_over_it():
    occ = Occluders.from_geometry(shelf_room())
    assert occ.blocked(np.array([6.0, 4.0, 1.5]), np.array([2.0, 4.0, 1.5]))
    assert not occ.blocked(np.array([6.0, 4.0, 1.6]), np.array([2.0, 4.0, 1.6]))
    assert occ.on_enclosure(np.array([4.0, 4.0, 0.0002]), 2e-3)


def test_a_box_on_a_tabletop_does_not_diffract_along_its_bottom():
    top = np.array([[3, 1, 0.75], [7, 1, 0.75], [7, 7, 0.75], [3, 7, 0.75]], dtype=float)
    v, f = box(np.array([4.0, 2.0, 0.7502]), np.array([4.6, 6.0, 1.5]))
    edges = obstacle_edges(room_with(np.vstack([top, v]), [(0, 1, 2, 3)] + [
        tuple(4 + i for i in q) for q in f]))
    wedges = [e for e in edges if e.theta_w == pytest.approx(1.5 * np.pi)]
    assert len(wedges) == 8
    assert not any(np.allclose([e.start[2], e.end[2]], 0.7502) for e in wedges)
    assert sum(e.theta_w == pytest.approx(2 * np.pi) for e in edges) == 4


# The wedge zones against the geometry they stand for.
#
# The reflection zone of a wedge face is where the specular reflection off that
# face exists (Kirsch and Ewert 2023, Sec. II; it decides the polarity of the
# theta_+ term of eq. 2). Checked against the reflection itself: the image of
# the source in the face plane, and the point where its line to the receiver
# meets the face.


def point(theta, r):
    """Open angle theta from face A (+x), rotating away from the solid (via -y)."""
    return np.array([r * np.cos(theta), -r * np.sin(theta), 0.0])


def reflection_exists(theta_w, s, r):
    for u in (np.array([1.0, 0.0, 0.0]), np.array([np.cos(theta_w), -np.sin(theta_w), 0.0])):
        n = np.cross([0.0, 0.0, 1.0], u)
        if (s @ n) * (r @ n) <= 0.0:
            continue
        image = s - 2.0 * (s @ n) * n
        d = r - image
        t = -(image @ n) / (d @ n)
        if 0.0 < t < 1.0 and (image + t * d) @ u > 0.0:
            return True
    return False


def test_reflection_zone_is_where_the_reflection_exists():
    rng = np.random.default_rng(5)
    margin = np.deg2rad(1.5)
    checked = 0
    for _ in range(2000):
        theta_w = 2 * np.pi if rng.random() < 0.5 else rng.uniform(4 * np.pi / 3, 2 * np.pi)
        ts, tr = rng.uniform(0.0, theta_w, 2)
        near = (abs(ts - tr) - np.pi, ts + tr - np.pi, ts + tr - 2 * theta_w + np.pi,
                ts, theta_w - ts, tr, theta_w - tr)
        if min(abs(x) for x in near) < margin:
            continue
        s, r = point(ts, rng.uniform(0.5, 5.0)), point(tr, rng.uniform(0.5, 5.0))
        edge = Edge([0, 0, -2], [0, 0, 2], [1, 0, 0], [np.cos(theta_w), -np.sin(theta_w), 0], theta_w)
        zone = wedge_path(edge, s, r).zone
        assert (zone == "reflection") == reflection_exists(theta_w, s, r), (theta_w, ts, tr)
        checked += 1
    assert checked > 1500


def panel_room(split_floor=False, noise=None):
    """A 10 x 8 x 4 m room with a 2 x 2 m panel at x = 5 (a flat object,
    Kirsch and Ewert 2023, Sec. VI); optionally the floor split along y = 4."""
    v = [ROOM_V]
    faces = [f for f in ROOM_F if not (split_floor and f == (0, 1, 2, 3))]
    names = ["wall"] * 4 + ["floor", "ceiling"]
    if split_floor:
        names = ["wall"] * 4 + ["ceiling", "floor", "floor"]
        seam = np.array([[0.0, 4.0, 0.0], [10.0, 4.0, 0.0]])
        if noise is not None:
            seam = seam + noise
        v.append(seam)
        faces += [(0, 1, 9, 8), (8, 9, 2, 3)]
    n = sum(len(x) for x in v)
    v.append(PANEL)
    faces.append(tuple(range(n, n + 4)))
    names.append("panel")
    materials = {name: Material(a * TILT, BANDS.copy()) for name, a in
                 (("wall", 0.1), ("floor", 0.05), ("ceiling", 0.3), ("panel", 0.1))}
    return MeshGeometry(vertices=np.vstack(v), faces=tuple(faces), face_materials=tuple(names),
                        materials=materials, obstacles=("panel",))


def spec_at(source, receiver, geometry=None, **kw):
    return RoomSpec(geometry=panel_room() if geometry is None else geometry,
                    source=np.asarray(source, float), receiver=np.asarray(receiver, float),
                    air_absorption=False, **kw)


def direct(paths):
    return [p for p in paths if p.order == 0]


def test_an_occluded_pair_diffracts_over_all_four_edges():
    s, r = np.array([2.0, 4.0, 1.5]), np.array([8.0, 4.0, 1.5])
    paths = direct(diffraction_paths(spec_at(s, r), max_order=0))
    assert len(paths) == 4 and all(p.zone == "shadow" for p in paths)
    apex = np.array([5.0, 4.0, TOP])
    # the panel is centred on the line, so all four edges are equally far
    top = [p for p in paths if np.allclose(p.doa, apex - r)]
    assert len(top) == 1
    assert top[0].distance == pytest.approx(np.linalg.norm(s - apex) + np.linalg.norm(apex - r))
    assert top[0].delay == pytest.approx(top[0].distance / C)


def test_grazing_pair_is_in_the_view_zone_of_the_top_edge():
    z = TOP + 0.023
    r = np.array([8.0, 4.0, z])
    paths = direct(diffraction_paths(spec_at([2.0, 4.0, z], r), max_order=0))
    top = [p for p in paths if np.allclose(p.doa, np.array([5.0, 4.0, TOP]) - r)]
    assert len(top) == 1 and top[0].zone == "view"


def floor_paths(paths, before, after):
    gain = np.sqrt(1 - 0.05 * TILT)
    return [round(p.distance, 6) for p in paths if p.order == 1
            and min(abs(p.distance - before), abs(p.distance - after)) < 1e-6
            and np.allclose(p.band_gain, gain, rtol=1e-6)]


def expected_floor(s, r):
    apex = np.array([5.0, 4.0, TOP])
    floor = np.array([1.0, 1.0, -1.0])
    before = np.linalg.norm(s * floor - apex) + np.linalg.norm(apex - r)
    after = np.linalg.norm(s - apex) + np.linalg.norm(apex - r * floor)
    return before, after


def test_floor_paths_reflect_before_and_after_the_diffraction():
    s, r = np.array([2.0, 4.0, 1.5]), np.array([8.0, 4.0, 1.0])
    before, after = expected_floor(s, r)
    found = floor_paths(diffraction_paths(spec_at(s, r), max_order=1), before, after)
    assert sorted(found) == sorted([round(before, 6), round(after, 6)])


def test_a_reflection_on_a_floor_seam_counts_once():
    # Both floor reflections over the top edge land on y = 4, the seam.
    # Nanometre noise on the seam's ends, as CAD exports carry, makes the two
    # faces' planes differ in the last bits; one image per face then counted
    # the path twice or dropped it (8 and 1 of 30 trials before the images
    # were built per plane).
    s, r = np.array([2.0, 4.0, 1.5]), np.array([8.0, 4.0, 1.0])
    before, after = expected_floor(s, r)
    expected = sorted([round(before, 6), round(after, 6)])
    rng = np.random.default_rng(1)
    for _ in range(30):
        g = panel_room(split_floor=True, noise=rng.normal(0.0, 1e-9, (2, 3)))
        found = floor_paths(diffraction_paths(spec_at(s, r, g), max_order=1), before, after)
        assert sorted(found) == expected


def test_only_the_legs_at_the_apex_pass_the_edge_faces():
    # source -> wall x = 10 -> top edge -> receiver: the source's leg to the
    # wall crosses the panel at z = 1.59, so the path is blocked; skipping the
    # edge's face on every leg let it through
    s, r = np.array([4.0, 4.0, 1.5]), np.array([2.0, 4.0, 1.5])
    apex = np.array([5.0, 4.0, TOP])
    image = s * np.array([-1.0, 1.0, 1.0]) + np.array([20.0, 0.0, 0.0])
    through = np.linalg.norm(image - apex) + np.linalg.norm(apex - r)
    paths = diffraction_paths(spec_at(s, r), max_order=1)
    assert not any(abs(p.distance - through) < 1e-6 for p in paths)


def test_occlusion_agrees_with_pyroomacoustics_visibility():
    # occluded exactly when the receiver is in the shadow zone of all four edges
    rng = np.random.default_rng(0)
    source = np.array([2.0, 4.0, 1.5])
    for _ in range(40):
        receiver = np.array([rng.uniform(6.0, 9.0), rng.uniform(2.5, 5.5), rng.uniform(0.8, 3.0)])
        spec = spec_at(source, receiver)
        # pyroomacoustics returns one band at max_order = 0 on a mesh: search to 1
        visible = bool(np.any(image_sources(spec, 1).order == 0))
        paths = direct(diffraction_paths(spec, max_order=0))
        shadowed = len(paths) == 4 and all(p.zone == "shadow" for p in paths)
        assert visible != shadowed, receiver


FS = 48000.0


def flat(f):
    return np.ones(np.shape(f), dtype=complex)


def test_a_flat_path_renders_exactly_like_an_image_source():
    doa, d, g = np.array([1.0, -2.0, 0.5]), 3.7, np.linspace(0.9, 0.5, 8)
    path = DiffractionPath(d / C, d, doa, g, flat, 0, "view", 0)
    iss = ImageSourceSet(delay=np.array([d / C]), gain=(g / (4 * np.pi * d))[None, :],
                         doa=pf.Coordinates(*doa), order=np.array([0]), bands=BANDS.copy(), fs=FS)
    np.testing.assert_allclose(encode_diffraction([path], 2, 4096, FS), encode(iss, 2, 4096), atol=1e-14)


def test_rendering_is_linear_in_the_transfer():
    doa = np.array([0.0, 1.0, 0.0])
    one = DiffractionPath(0.01, 3.43, doa, np.ones(8), flat, 0, "view", 0)
    two = DiffractionPath(0.01, 3.43, doa, np.ones(8), lambda f: 2 * flat(f), 0, "view", 0)
    np.testing.assert_allclose(encode_diffraction([two], 1, 2048, FS),
                               2 * encode_diffraction([one], 1, 2048, FS), atol=1e-14)


def test_diffraction_doas_rotate_like_image_sources():
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    s, r = [2.0, 4.0, 1.5], [8.0, 4.0, 1.5]
    plain = diffraction_paths(spec_at(s, r), max_order=0)
    turned = diffraction_paths(spec_at(s, r, receiver_rotation=rot), max_order=0)
    for a, b in zip(plain, turned):
        np.testing.assert_allclose(b.doa, rot.T @ a.doa, atol=1e-12)


def test_no_paths_render_silence():
    assert not np.any(encode_diffraction([], 1, 1024, FS))


def window_db(rir, r):
    n0 = r / C * FS
    a, b = int(n0 - 0.3e-3 * FS), int(n0 + 0.6e-3 * FS)
    return 10 * np.log10(np.sum(rir[a:b, 0] ** 2))


def test_diffraction_off_changes_nothing_and_on_adds_the_paths():
    spec = RoomSpec(geometry=panel_room(), source=np.array([2.0, 4.0, 1.5]),
                    receiver=np.array([8.0, 4.0, 3.0]), air_absorption=False)
    off = render_ambisonics(prepare(spec, image_sources(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2), normalization="N3D")
    on = render_ambisonics(prepare(spec, image_sources(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2, diffraction=True), normalization="N3D")
    n = off.early.shape[0]
    added = encode_diffraction(diffraction_paths(spec, max_order=1), 1, n, spec.fs)
    np.testing.assert_allclose(on.early - off.early, added, atol=1e-12)
    np.testing.assert_array_equal(on.late, off.late)


def test_the_field_is_continuous_across_the_panel_shadow_boundary():
    # boundary over the top edge (z = 2.5) at x = 8: z = 1.5 + 1.0 * 6 / 3 = 3.5
    levels = {}
    for flag in (False, True):
        for z in (3.495, 3.505):
            spec = RoomSpec(geometry=panel_room(), source=np.array([2.0, 4.0, 1.5]),
                            receiver=np.array([8.0, 4.0, z]), air_absorption=False)
            rir = render_ambisonics(prepare(spec, image_sources(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2, diffraction=flag)).early
            levels[flag, z] = window_db(rir, float(np.linalg.norm(spec.receiver - spec.source)))
    assert abs(levels[False, 3.505] - levels[False, 3.495]) > 6.0
    assert abs(levels[True, 3.505] - levels[True, 3.495]) < 0.5


def box_geometry():
    return room_with(*box(np.array([4.0, 3.0, 0.0]), np.array([4.6, 5.0, 1.5])))


def line_distance(e, p):
    ends, _ = toolbox_frame(e)
    return np.linalg.norm(np.cross(ends[1] - ends[0], ends[0] - p)) / e.length


def test_toolbox_cyl_puts_face_b_at_theta_w_and_the_solid_beyond():
    for e in obstacle_edges(box_geometry()):
        ends, _ = toolbox_frame(e)
        mid = ends.mean(axis=0)
        pts = np.array([mid + 0.1 * e.face_a, mid + 0.1 * e.face_b, mid + 0.05 * e.face_a + 0.05 * e.face_b])
        _, t, z = toolbox_cyl(e, pts).T
        np.testing.assert_allclose(t[:2], [0.0, e.theta_w], atol=1e-9)
        assert t[2] > e.theta_w
        np.testing.assert_allclose(z, e.length / 2.0, atol=1e-12)


def test_toolbox_cyl_on_a_knife_edge_runs_from_zero_to_two_pi():
    for e in obstacle_edges(panel_room()):
        ends, _ = toolbox_frame(e)
        mid = ends.mean(axis=0)
        on = toolbox_cyl(e, mid + 0.1 * e.face_a)[0, 1]
        np.testing.assert_allclose(on, 0.0, atol=1e-9)
        for side in (1.0, -1.0):
            t = toolbox_cyl(e, mid + 0.1 * e.face_a + side * 1e-3 * np.cross(e.axis, e.face_a))[0, 1]
            assert 0.0 <= t < 0.02 or 2 * np.pi - 0.02 < t <= 2 * np.pi
        assert {toolbox_cyl(e, mid + 0.1 * e.face_a + s * 1e-3 * np.cross(e.axis, e.face_a))[0, 1] < 1.0
                for s in (1.0, -1.0)} == {True, False}


def test_segments_cover_the_edge_within_five_degrees():
    e = obstacle_edges(box_geometry())[0]
    s, r = np.array([2.0, 4.0, 1.0]), np.array([8.0, 4.0, 1.0])
    seg = edge_segments(e, s, r)
    assert seg[0, 0] == 0.0 and np.isclose(seg[-1, 1], e.length)
    np.testing.assert_allclose(seg[1:, 0], seg[:-1, 1])
    near = min(line_distance(e, p) for p in (s, r))
    assert np.max(np.diff(seg, axis=1)) <= 0.087 * near + 1e-12


def test_the_two_top_edges_of_a_box_run_along_its_top_face():
    edges = obstacle_edges(box_geometry())
    top = [e for e in edges if np.isclose(e.start[2], 1.5) and np.isclose(e.end[2], 1.5)
           and abs(e.axis[1]) > 1 - 1e-9]
    assert len(top) == 2
    assert along_plane(top[0], top[1])
    verticals = [e for e in edges if abs(e.axis[2]) > 1 - 1e-9]
    shares = [v for v in verticals if set(top[0].faces) & set(v.faces)]
    apart = [v for v in verticals if not set(top[0].faces) & set(v.faces)]
    assert shares and apart
    assert along_plane(top[0], shares[0])
    assert not along_plane(top[0], apart[0])


def rotated_box_geometry(angles):
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("zyx", angles, degrees=True).as_matrix()
    v, f = box(np.array([-0.3, -1.0, -0.75]), np.array([0.3, 1.0, 0.75]))
    return room_with(v @ rot.T + np.array([5.0, 4.0, 2.0]), f), rot


def test_coplanar_edges_of_a_rotated_box_are_a_pair_at_every_point():
    t = np.linspace(0.05, 0.95, 7)
    for angles in ([31.0, 17.0, -23.0], [-12.0, 40.0, 65.0], [77.0, -33.0, 8.0]):
        geometry, rot = rotated_box_geometry(angles)
        y = rot @ np.array([0.0, 1.0, 0.0])
        for face in (1, 0, 2, 3, 4, 5):
            pair = [e for e in obstacle_edges(geometry)
                    if face in e.faces and abs(e.axis @ y) > 1 - 1e-9]
            if len(pair) != 2:
                continue
            assert along_plane(pair[0], pair[1])
            for e1, e2 in (pair, pair[::-1]):
                for a in t:
                    for b in t:
                        p1 = e1.start + a * (e1.end - e1.start)
                        p2 = e2.start + b * (e2.end - e2.start)
                        assert pair_valid(e1, e2, p1, p2)


def test_segments_are_floored_when_a_point_is_on_the_edge_line():
    e = obstacle_edges(box_geometry())[0]
    near = toolbox_frame(e)[0][0] + 0.3 * e.axis + 1e-4 * np.cross(e.axis, e.face_a)
    seg = edge_segments(e, near, np.array([8.0, 4.0, 1.0]))
    assert len(seg) <= np.ceil(e.length / 0.01)


def test_an_edge_behind_another_is_not_a_pair():
    edges = obstacle_edges(box_geometry())
    e1 = edges[0]
    inside = toolbox_frame(e1)[0].mean(axis=0) + 0.1 * e1.face_a + 0.1 * e1.face_b
    assert not pair_valid(e1, edges[1], toolbox_frame(e1)[0].mean(axis=0), inside)


def test_btms_segments_of_a_free_edge_sum_to_the_kernel_on_the_whole_edge():
    s, r = np.array([2.0, 4.0, 1.5]), np.array([8.0, 4.0, 3.0])
    spec = RoomSpec(geometry=panel_room(), source=s, receiver=r, air_absorption=False, speed_of_sound=C)
    paths = btms_paths(spec, max_order=0, second_order=False)
    assert paths and all(p.order == 1 for p in paths)
    edges = obstacle_edges(spec.geometry)
    top = next(k for k, e in enumerate(edges) if np.isclose(e.start[2], 2.5) and np.isclose(e.end[2], 2.5))
    part = [p for p in paths if p.edges == (top,)]
    e = edges[top]
    assert len(part) > 1 and len(part) == len(edge_segments(e, s, r))
    total = np.zeros(4800)
    for p in part:
        total[p.start:p.start + p.ir.size] += p.ir
    (rs, ts, zs), (rr, tr, zr) = toolbox_cyl(e, np.array([s, r]))
    whole, *_ = wedge1st_ir(spec.fs, C, 2 * np.pi - e.theta_w, rs, ts, zs, rr, tr, zr, np.array([0.0, e.length]))
    # the analytic apex approximation (zrelmax = 0.1) is good to ~1.6e-5 of peak and its range moves with the cut
    np.testing.assert_allclose(total[:whole.size], whole, atol=5e-5 * np.max(np.abs(whole)))


def _btms_total(a, b):
    spec = RoomSpec(geometry=panel_room(), source=a, receiver=b, air_absorption=False, speed_of_sound=C)
    return encode_btms(btms_paths(spec, max_order=1, second_order=False), 0, 4800, spec.fs)[:, 0]


def test_btms_is_reciprocal():
    s, r = np.array([2.0, 4.1, 1.5]), np.array([8.0, 4.0, 3.1])
    total = _btms_total(s, r)
    np.testing.assert_allclose(_btms_total(r, s), total, atol=1e-10 * np.max(np.abs(total)))


def test_btms_reciprocity_with_a_segment_boundary_on_the_apex_is_rounding_limited():
    s, r = np.array([2.0, 4.0, 1.5]), np.array([8.0, 4.0, 3.0])
    total = _btms_total(s, r)
    assert np.all(np.isfinite(total))
    np.testing.assert_allclose(_btms_total(r, s), total, atol=1e-6 * np.max(np.abs(total)))


def test_btms_contributions_after_the_render_length_are_dropped():
    spec = RoomSpec(geometry=panel_room(), source=np.array([2.0, 4.0, 1.5]),
                    receiver=np.array([8.0, 4.0, 3.0]), air_absorption=False, speed_of_sound=C)
    out = encode_btms(btms_paths(spec, max_order=0, second_order=False), 0, 10, spec.fs)
    assert out.shape == (10, 1) and np.all(out == 0.0)


def test_btms_skips_a_receiver_on_an_edge_line():
    geo = panel_room()
    e = obstacle_edges(geo)[0]
    on_line = e.start + 0.5 * (e.end - e.start)
    spec = RoomSpec(geometry=geo, source=np.array([2.0, 4.0, 1.5]), receiver=on_line,
                    air_absorption=False, speed_of_sound=C)
    out = encode_btms(btms_paths(spec, max_order=0, second_order=False), 1, 4800, spec.fs)
    assert np.all(np.isfinite(out))


def shelf_spec(s, r):
    return RoomSpec(geometry=box_geometry(), source=s, receiver=r, air_absorption=False, speed_of_sound=C)


def test_second_order_reaches_over_a_box_top():
    s, r = np.array([2.0, 4.0, 1.45]), np.array([7.0, 4.0, 1.45])
    paths = btms_paths(shelf_spec(s, r), max_order=0)
    second = [p for p in paths if p.order == 2]
    assert second
    assert all(len(p.edges) == 2 and p.edges[0] != p.edges[1] for p in second)


def test_second_order_is_reciprocal():
    s, r = np.array([2.0, 4.0, 1.45]), np.array([7.0, 4.0, 1.45])
    a = encode_btms([p for p in btms_paths(shelf_spec(s, r), max_order=0) if p.order == 2], 0, 4800, 48000.0)
    b = encode_btms([p for p in btms_paths(shelf_spec(r, s), max_order=0) if p.order == 2], 0, 4800, 48000.0)
    np.testing.assert_allclose(a, b, atol=1e-9 * np.max(np.abs(a)))


def test_second_order_needs_direct_paths_only():
    s, r = np.array([2.0, 4.0, 1.45]), np.array([7.0, 4.0, 1.45])
    assert all(p.order == 1 for p in btms_paths(shelf_spec(s, r), max_order=1, second_order=False))
    count = [sum(p.order == 2 for p in btms_paths(shelf_spec(s, r), max_order=m)) for m in (0, 1)]
    assert count[0] == count[1] > 0


def test_second_order_start_is_a_lower_bound_of_the_path_through_two_long_segments():
    s, r = np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])
    seg1 = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 6.0]])
    seg2 = np.array([[8.0, 1.0, -4.0], [8.0, 1.0, 2.0]])
    t = np.linspace(0.0, 1.0, 601)
    x = seg1[0] + t[:, None] * (seg1[1] - seg1[0])
    y = seg2[0] + t[:, None] * (seg2[1] - seg2[0])
    paths = (np.linalg.norm(x - s, axis=1)[:, None] + np.linalg.norm(x[:, None] - y[None], axis=2)
             + np.linalg.norm(y - r, axis=1)[None, :])
    true_min = paths.min()
    ends = [seg1[0], seg1[1], seg1.mean(axis=0)], [seg2[0], seg2[1], seg2.mean(axis=0)]
    brief = min(np.linalg.norm(a - s) + np.linalg.norm(b - a) + np.linalg.norm(r - b)
                for a in ends[0] for b in ends[1])
    assert brief > true_min + C / FS
    assert second_order_start(s, r, seg1, seg2, C, FS) <= int(np.floor(true_min / C * FS))


def test_second_order_runs_when_the_segments_are_long():
    s, r = np.array([1.0, 1.0, 1.45]), np.array([9.0, 7.0, 1.45])
    out = btms_paths(shelf_spec(s, r), max_order=0)
    assert any(p.order == 2 for p in out)
    assert all(p.start >= 0 and np.all(np.isfinite(p.ir)) for p in out)


def test_second_order_segment_pairs_sum_to_the_whole_edge_pair():
    s, r = np.array([2.0, 4.0, 1.45]), np.array([7.0, 4.0, 1.45])
    spec = shelf_spec(s, r)
    edges = obstacle_edges(spec.geometry)
    top = [k for k, e in enumerate(edges) if np.isclose(e.start[2], 1.5) and np.isclose(e.end[2], 1.5)
           and abs(e.axis[1]) > 1 - 1e-9]
    i, j = sorted(top, key=lambda k: edges[k].start[0])
    e1, e2 = edges[i], edges[j]
    part = [p for p in btms_paths(spec, max_order=0) if p.edges == (i, j)]
    assert len(part) > 1
    n = 4800
    total = np.zeros(n)
    for p in part:
        total[p.start:p.start + p.ir.size] += p.ir[:n - p.start]
    ends1, ends2 = toolbox_frame(e1)[0], toolbox_frame(e2)[0]
    steps, dz = element_grid(np.array([e1.length, e2.length]), C, FS)
    lower = max(np.linalg.norm(s - r), segment_distance(s, *ends1) + segment_distance(r, *ends2))
    start = int(np.floor(lower / C * FS))
    whole = wedge2nd_ir(toolbox_cyl(e1, s[None])[0], toolbox_cyl(e2, r[None])[0],
                        toolbox_cyl(e1, ends2), toolbox_cyl(e2, ends1), np.pi / np.array([e1.theta_w, e2.theta_w]),
                        np.array([[0.0, e1.length], [0.0, e2.length]]), dz, steps, along_plane(e1, e2),
                        start * C / FS, C, FS)
    ref = np.zeros(n)
    ref[start:start + whole.size] = whole[:n - start]
    db = 10 * np.log10(np.sum(total**2) / np.sum(ref**2))
    assert abs(db) < 0.1


def test_second_order_is_invariant_under_rotation_with_corner_sharing_pairs():
    # the segment of edge 2 touching the shared corner has an end on edge 1's line (r = 0), where theta is
    # undefined; the pair's angle must come from the common face, not from that end
    centre, half = np.array([5.0, 4.0, 2.0]), np.array([0.5, 0.3, 0.4])
    s0, r0 = np.array([-1.6, 0.05, 0.05]), np.array([1.7, -0.05, 0.2])

    def rot_z(deg):
        a = np.radians(deg)
        return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])

    def response(deg):
        # the box turned by deg, source and receiver turned by deg - 30: one configuration, two frames
        v, f = box(-half, half)
        spec = RoomSpec(geometry=room_with(v @ rot_z(deg).T + centre, f), source=centre + rot_z(deg - 30) @ s0,
                        receiver=centre + rot_z(deg - 30) @ r0, air_absorption=False, speed_of_sound=C)
        second = [p for p in btms_paths(spec, max_order=0) if p.order == 2]
        return encode_btms(second, 0, 4800, spec.fs)[:, 0]

    ref = response(30.0)
    np.testing.assert_allclose(response(0.0), ref, atol=1e-9 * np.max(np.abs(ref)))


def test_second_order_over_a_free_panel_is_invariant_under_rotation():
    # knife edges: the panel is both faces, so a computed angle of a coplanar edge is 0 or 2 pi - eps by
    # rounding; the toolbox (EDed2geo.m) gives it the reference plane's 0
    from scipy.spatial.transform import Rotation

    centre = np.array([5.0, 4.0, 2.0])
    v = np.array([[-0.5, -0.4, 0.0], [0.5, -0.4, 0.0], [0.5, 0.4, 0.0], [-0.5, 0.4, 0.0]])
    s0, r0 = np.array([-0.2, -0.05, 0.6]), np.array([0.25, 0.05, -0.5])

    def response(rot):
        spec = RoomSpec(geometry=room_with(v @ rot.T + centre, [(0, 1, 2, 3)], name="panel"),
                        source=centre + rot @ s0, receiver=centre + rot @ r0, air_absorption=False, speed_of_sound=C)
        second = [p for p in btms_paths(spec, max_order=0) if p.order == 2]
        assert second
        return encode_btms(second, 0, 4800, spec.fs)[:, 0]

    ref = response(np.eye(3))
    turned = Rotation.from_rotvec(np.radians(30.0) * np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)).as_matrix()
    np.testing.assert_allclose(response(turned), ref, atol=1e-9 * np.max(np.abs(ref)))


def split_top_box():
    """The box of box_geometry with its top split into two coplanar polygons
    along x = 4.3, so that neither long top edge touches both; the side faces
    carry the extra vertices."""
    v, f = box(np.array([4.0, 3.0, 0.0]), np.array([4.6, 5.0, 1.5]))
    v = np.vstack([v, [[4.3, 3.0, 1.5], [4.3, 5.0, 1.5]]])
    f = [f[0], (4, 8, 9, 7), (8, 5, 6, 9), (0, 1, 5, 8, 4), f[3], (2, 3, 7, 9, 6), f[5]]
    return room_with(v, f)


def long_top_edges(geometry, z):
    edges = obstacle_edges(geometry)
    return [e for e in edges if np.isclose(e.start[2], z) and np.isclose(e.end[2], z)
            and abs(e.axis[1]) > 1 - 1e-9]


def test_a_top_face_split_into_polygons_is_still_one_plane():
    top = long_top_edges(split_top_box(), 1.5)
    assert len(top) == 2
    assert along_plane(top[0], top[1])


def test_coplanar_tops_of_two_separate_boxes_are_not_along_a_plane():
    v1, f1 = box(np.array([4.0, 1.0, 0.0]), np.array([4.6, 3.0, 1.5]))
    v2, f2 = box(np.array([4.0, 5.0, 0.0]), np.array([4.6, 7.0, 1.5]))
    geometry = room_with(np.vstack([v1, v2]), f1 + [tuple(8 + i for i in q) for q in f2])
    first = [e for e in long_top_edges(geometry, 1.5) if e.component == 0]
    second = [e for e in long_top_edges(geometry, 1.5) if e.component == 1]
    assert len(first) == len(second) == 2
    assert along_plane(*first) and along_plane(*second)
    assert not along_plane(first[0], second[0]) and not along_plane(first[1], second[1])


def test_second_order_between_collinear_edges_is_skipped_without_a_warning():
    v = np.array([[5, 1, 0.5], [5, 3, 0.5], [5, 3, 2.5], [5, 1, 2.5],
                  [5, 5, 0.5], [5, 7, 0.5], [5, 7, 2.5], [5, 5, 2.5]], dtype=float)
    geometry = room_with(v, [(0, 1, 2, 3), (4, 5, 6, 7)], name="panel")
    spec = RoomSpec(geometry=geometry, source=np.array([2.0, 4.0, 3.2]), receiver=np.array([8.0, 4.0, 3.2]),
                    air_absorption=False, speed_of_sound=C)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = btms_paths(spec, max_order=0)
    edges = obstacle_edges(geometry)
    top = [k for k, e in enumerate(edges) if np.isclose(e.start[2], 2.5) and np.isclose(e.end[2], 2.5)]
    assert len(top) == 2
    assert not any(p.edges in ((top[0], top[1]), (top[1], top[0])) for p in out)
    assert all(np.all(np.isfinite(p.ir)) for p in out)


def test_true_and_udfa_are_the_same_model():
    spec = RoomSpec(geometry=panel_room(), source=np.array([2.0, 4.0, 1.5]),
                    receiver=np.array([8.0, 4.0, 3.0]), air_absorption=False)
    a = render_ambisonics(prepare(spec, image_sources(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2, diffraction=True))
    b = render_ambisonics(prepare(spec, image_sources(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2, diffraction="udfa"))
    np.testing.assert_array_equal(a.rir, b.rir)


def test_btms_mode_adds_exactly_the_btms_field():
    spec = RoomSpec(geometry=panel_room(), source=np.array([2.0, 4.0, 1.5]),
                    receiver=np.array([8.0, 4.0, 3.0]), air_absorption=False)
    off = render_ambisonics(prepare(spec, image_sources(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2), normalization="N3D")
    on = render_ambisonics(prepare(spec, image_sources(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2, diffraction="btms"), normalization="N3D")
    added = encode_btms(btms_paths(spec, max_order=1), 1, off.early.shape[0], spec.fs)
    np.testing.assert_allclose(on.early - off.early, added, atol=1e-12)
    np.testing.assert_array_equal(on.late, off.late)


def test_an_unknown_diffraction_model_is_refused():
    spec = RoomSpec(geometry=panel_room(), source=np.array([2.0, 4.0, 1.5]),
                    receiver=np.array([8.0, 4.0, 3.0]))
    with pytest.raises(ValueError, match="diffraction"):
        render_ambisonics(prepare(spec, image_sources(spec, 1), RoomTarget.from_room(spec), max_order=0, duration_s=0.1, diffraction="btm"))


def edverify_plate():
    """The 1 x 2 m plate of EDverify.m tests 2 and 3 (plane y = 0, edge x = 0,
    -1 <= z <= 1) in a 400 m enclosure that absorbs everything."""
    big = 200.0 * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    plate = np.array([[0, 0, 1], [0, 0, -1], [-1, 0, -1], [-1, 0, 1]], float)
    mats = {"absorber": Material(np.linspace(0.999, 1.0, 8), BANDS.copy()),
            "plate": Material(np.linspace(0.0, 1e-6, 8), BANDS.copy())}
    return MeshGeometry(np.vstack([big, plate]), tuple(ROOM_F) + ((8, 9, 10, 11),),
                        ("absorber",) * 6 + ("plate",), mats, obstacles=("plate",))


@pytest.mark.parametrize("z", [0.0, 1.0], ids=["perpendicular", "skewed"])
def test_continuity_across_zone_boundaries_as_edverify(z):
    """EDverify.m tests 2 and 3: at 100 Hz the field exactly on the direct-sound
    and the specular zone boundary equals the mean of the fields 0.001 deg either
    side to 1e-5. Direct sound and specular reflection are placed as
    EDmakefirstorderirs.m places them and counted off the boundary only, as the
    image sources are; on it the scaled component comes from btms_paths."""
    c, f = 344.0, 100.0
    geo = edverify_plate()
    s = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
    image = s * np.array([1.0, -1.0, 1.0])

    def at(src, r, fs):
        d = float(np.linalg.norm(r - src))
        slot = d / c * fs
        n = np.floor(slot)
        w = 2.0 * np.pi * f / fs
        return ((1.0 - (slot - n)) * np.exp(-1j * w * n) + (slot - n) * np.exp(-1j * w * (n + 1))) / d

    def crossing_x(src, r):
        # x where the line meets the plate's plane (NaN if it does not); the plate spans -1 < x < 0
        if src[1] * r[1] >= 0.0:
            return np.nan
        return src[0] + src[1] / (src[1] - r[1]) * (r[0] - src[0])

    def fields(r):
        spec = RoomSpec(geometry=geo, source=s, receiver=r, air_absorption=False, speed_of_sound=c)
        diff = sum(np.sum(p.ir * np.exp(-2j * np.pi * f * (p.start + np.arange(p.ir.size)) / spec.fs))
                   * p.band_gain[1] for p in btms_paths(spec, max_order=0, second_order=False))
        direct = at(s, r, spec.fs) if crossing_x(s, r) > 1e-12 else 0.0
        specular = at(image, r, spec.fs) if crossing_x(image, r) < -1e-12 else 0.0
        return direct + diff, specular + diff

    def relerr(values):
        mean = (abs(values[0]) + abs(values[2])) / 2.0
        return abs((mean - abs(values[1])) / mean)

    def on_circle(deg):
        return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), z])

    # the boundary receivers exactly: -s for the direct sound, the image mirrored for the specular
    direct = [fields(r)[0] for r in (on_circle(315.001), np.array([-s[0], -s[1], z]), on_circle(314.999))]
    specular = [fields(r)[1] for r in (on_circle(45.001), np.array([-image[0], -image[1], z]), on_circle(44.999))]
    assert relerr(direct) < 1e-5
    assert relerr(specular) < 1e-5


@pytest.mark.parametrize("mirror, scale", [((1, 1, 1), 2.0 / 3.0), ((1, -1, 1), 1.0 / 3.0)],
                         ids=["direct", "specular"])
def test_a_path_through_a_corner_is_scaled_once_as_edfindconvexgapaths(mirror, scale):
    # the path through the plate's corner (0, 0, 1) lies on the zone boundaries of
    # both edges meeting there; EDfindconvexGApaths.m scales the direct sound to 2/3
    # and a specular reflection to 1/3 for such a corner hit
    s = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
    origin = s * np.array(mirror)
    r = np.array([0.0, 0.0, 2.0]) - origin
    spec = RoomSpec(geometry=edverify_plate(), source=s, receiver=r, air_absorption=False, speed_of_sound=344.0)
    paths = btms_paths(spec, max_order=0, second_order=False)
    assert sum(p.singular for p in paths if p.order == 1) == 2
    scaled = [p for p in paths if p.order == 0]
    assert len(scaled) == 1
    np.testing.assert_allclose(scaled[0].ir.sum(), scale / np.linalg.norm(r - origin), rtol=1e-12)


def test_a_receiver_on_a_zone_boundary_gets_half_the_geometric_sound():
    # source, the panel's top edge (z = 2.5 at x = 5) and receiver exactly collinear:
    # the receiver lies on the incident shadow boundary of that edge. The image
    # sources drop the direct sound there and the singular BTMS term is zeroed, so
    # without the half direct sound the field would lose half its pressure. Compared
    # at 100 Hz as EDverify.m does, with the neighbour 0.1 mm into the shadow (on the
    # lit side pyroomacoustics drops the direct sound up to about 1 mm from the edge).
    s, r = np.array([2.0, 4.0, 1.5]), np.array([8.0, 4.0, 3.5])

    def at_100_hz(receiver):
        spec = RoomSpec(geometry=panel_room(), source=s, receiver=receiver, air_absorption=False)
        w = render_ambisonics(prepare(spec, image_sources(spec, 1), RoomTarget.from_room(spec), max_order=0, duration_s=0.1, diffraction="btms")).early[:, 0]
        return abs(np.sum(w * np.exp(-2j * np.pi * 100.0 * np.arange(w.size) / spec.fs)))

    spec = RoomSpec(geometry=panel_room(), source=s, receiver=r, air_absorption=False)
    halves = [p for p in btms_paths(spec, max_order=0) if p.order == 0]
    assert len(halves) == 1 and halves[0].singular
    np.testing.assert_allclose(halves[0].ir.sum(), 0.5 / np.linalg.norm(r - s), rtol=1e-12)
    np.testing.assert_allclose(at_100_hz(r), at_100_hz(r - np.array([0.0, 0.0, 1e-4])), rtol=1e-3)
