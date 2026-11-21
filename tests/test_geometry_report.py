"""Geometry issues of compas_acoustics.geometry.analyze on synthetic scenes (ported from compas_soundscape).

Checks the user-facing warnings: outside sources, leaks + leaking holes,
blocked direct paths and separated rooms.
"""

from test_geometry_prepare import _box
from test_geometry_prepare import _room
from test_geometry_prepare import _Scene

from compas_acoustics.geometry import ReportOptions
from compas_acoustics.geometry import Seed
from compas_acoustics.geometry import analyze


def _analyze(scene, sources, receivers, **settings):
    seeds = [Seed(f"s{i}", "source", p) for i, p in enumerate(sources)]
    seeds += [Seed(f"r{i}", "receiver", p) for i, p in enumerate(receivers)]
    mesh = scene.prepare([s.position for s in seeds], **settings)
    pairs = [(s.id, r.id) for s in seeds if s.kind == "source" for r in seeds if r.kind == "receiver"]
    return mesh, pairs, analyze(mesh, seeds, pairs, ReportOptions(image_source_order=3, ray_tracing=True))


def _codes(analysis, severity=None):
    return {i.code for i in analysis.issues if severity is None or i.severity == severity}


def test_clean_box_has_no_errors_or_warnings():
    scene = _Scene()
    _room(scene)
    _, _, a = _analyze(scene, [[2, 2, 1.5]], [[4, 2, 1.5]])
    assert not _codes(a, "error")
    assert not _codes(a, "warning")


def test_outside_source_is_an_error():
    scene = _Scene()
    _room(scene)
    _, _, a = _analyze(scene, [[20, 2, 1.5]], [[4, 2, 1.5]])
    assert "seed_outside" in _codes(a, "error")


def test_missing_ceiling_triangle_leaks():
    scene = _Scene()
    v, f = _box(0, 0, 0, 6, 4, 3)
    del f[2]
    scene.add("room", v, f)
    _, _, a = _analyze(scene, [[2, 2, 1.5]], [[4, 2, 1.5]])
    assert "leaking_hole" in _codes(a, "error")
    assert "sound_leak" in _codes(a)
    assert any(a.loop_leaking)


def test_partial_wall_blocks_direct_path():
    scene = _Scene()
    _room(scene)
    cv, cf = _box(2.9, 1, 0, 3.1, 3, 3)  # thick wall, gap on both sides
    scene.add("divider", cv, cf)
    _, _, a = _analyze(scene, [[1.5, 2, 1.5]], [[4.5, 2, 1.5]])
    assert "direct_path_blocked" in _codes(a, "warning")
    assert "no_acoustic_path" not in _codes(a)
    assert a.blocked_paths and a.blocked_paths[0]["face_ids"]


def test_closed_rooms_have_no_acoustic_path():
    scene = _Scene()
    va, fa = _box(0, 0, 0, 3, 4, 3)
    vb, fb = _box(3.2, 0, 0, 6, 4, 3)
    scene.add("a", va, fa)
    scene.add("b", vb, fb)
    _, _, a = _analyze(scene, [[1.5, 2, 1.5]], [[4.5, 2, 1.5]])
    assert "no_acoustic_path" in _codes(a, "error")
