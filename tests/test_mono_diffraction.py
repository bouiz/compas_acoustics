"""Edge diffraction in the mono method."""

import warnings

import numpy as np
import pytest
from compas.datastructures import Mesh

import compas_acoustics as ca

pytest.importorskip("spharpy")  # compas_acoustics.hybrid.diffraction needs it

FS = 48000
WALL = ca.AcousticMaterial("wall", 0.3, 0.1)


def _panel_scene(receiver_y):
    """A 2 m wide, 3 m tall-room-high panel at x = 4 between the source and a receiver behind it."""
    scene = ca.Scene("shadow")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=WALL))
    panel = Mesh.from_vertices_and_faces([[4, 2, 0.0], [4, 4, 0.0], [4, 4, 3.0], [4, 2, 3.0]], [[0, 1, 2, 3]])
    scene.add_acoustic_object(ca.AcousticObject(panel, ca.AcousticMaterial("panel", 0.2, 0.1), name="panel", role="obstacle"))
    scene.add_source(ca.Source("s", 2, 3, 1.5))
    scene.add_receiver(ca.Receiver("r", 6, receiver_y, 1.5))
    return scene


def _direct_level(scene, diffraction, until=None):
    """Level in a window round the direct arrival (-0.3 to +0.6 ms), as the hybrid's continuity test reads it, or to a
    path length ``until``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ir = next(iter(scene.simulate(ca.Mono(image_source_order=1, ray_tracing=False, diffraction=diffraction)).values())).denormalized()
    n0 = np.linalg.norm(np.subtract(scene.receivers[0].position, scene.sources[0].position)) / 343.98 * FS
    end = n0 + 0.6e-3 * FS if until is None else until / 343.98 * FS + 0.6e-3 * FS
    return 10 * np.log10(np.sum(ir[int(n0 - 0.3e-3 * FS) : int(end)] ** 2) + 1e-30)


def test_diffraction_fills_the_shadow_and_keeps_the_level_continuous():
    # the shadow boundary of the edge at (4, 2): the line from the source through it reaches y = 1 at x = 6
    shadow, lit = _panel_scene(1.005), _panel_scene(0.995)
    jump_without = _direct_level(lit, False) - _direct_level(shadow, False)
    jump_with = _direct_level(lit, True) - _direct_level(shadow, True)
    assert jump_without > 6.0
    assert abs(jump_with) < 0.5
    # deep in the shadow the sound round the panel's edges (4.47 m, before the floor's 5 m) is there
    deep = _panel_scene(3.0)
    assert _direct_level(deep, True, until=4.6) > _direct_level(deep, False, until=4.6) + 10.0


def test_mono_diffraction_is_opt_in_and_recorded():
    scene = _panel_scene(3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = next(iter(scene.simulate(ca.Mono(image_source_order=1, ray_tracing=False)).values()))
        with_paths = next(iter(scene.simulate(ca.Mono(image_source_order=1, n_rays=2000, duration=0.5, diffraction=True)).values()))
    assert "diffraction" not in plain.generation and with_paths.generation["diffraction"] == "udfa"
    assert with_paths.ir.shape == (int(0.5 * FS),)
    with pytest.raises(ValueError, match="BTMS"):
        ca.Mono(diffraction="btms")
