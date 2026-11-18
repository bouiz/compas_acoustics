"""Mono's diffracted paths take a measured source's pattern, as the hybrid's do."""

import warnings

import numpy as np
import pyroomacoustics as pra
import pytest
from compas.datastructures import Mesh

import compas_acoustics as ca

pytest.importorskip("spharpy")

FS = 48000


def _facing_minus_x(back_gain):
    """A measured pattern, 1 in its front half and ``back_gain`` behind, turned to face -x (a pyroomacoustics object
    keeps its own orientation, whatever the source's direction)."""
    grid = pra.doa.GridSphere(n_points=300)
    irs = np.zeros((300, 1024))
    irs[:, 0] = np.where(grid.cartesian[0] >= 0.0, 1.0, back_gain)
    return pra.directivities.MeasuredDirectivity(pra.directivities.Rotation3D([0, 180], rot_order="yz"), grid, irs, FS)


def _shadowed(directivity, direction):
    scene = ca.Scene("shadow")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=ca.AcousticMaterial("wall", 0.3, 0.1)))
    panel = Mesh.from_vertices_and_faces([[4, 2, 0.0], [4, 4, 0.0], [4, 4, 3.0], [4, 2, 3.0]], [[0, 1, 2, 3]])
    scene.add_acoustic_object(ca.AcousticObject(panel, ca.AcousticMaterial("panel", 0.2, 0.1), name="panel", role="obstacle"))
    scene.add_source(ca.Source("s", 2, 3, 1.5, directivity=directivity, direction=direction))
    scene.add_receiver(ca.Receiver("r", 6, 3, 1.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return next(iter(scene.simulate(ca.Mono(image_source_order=0, ray_tracing=False, diffraction=True)).values())).denormalized()


def test_the_diffracted_sound_follows_the_source_pattern():
    """Behind a room-high panel only the diffracted paths arrive (image source order 0). They leave the source towards
    the panel's edges, in +x: a pattern facing -x, 0.1 behind, sends them 20 dB less than an omnidirectional source."""
    omni = _shadowed("omnidirectional", [1, 0, 0])
    away = _shadowed(_facing_minus_x(0.1), [-1, 0, 0])
    assert np.sum(omni**2) > 0
    assert 10 * np.log10(np.sum(away**2) / np.sum(omni**2)) == pytest.approx(-20.0, abs=0.5)
