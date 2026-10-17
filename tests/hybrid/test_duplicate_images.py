"""An image reached through an edge is one image.

Where the path to an image passes exactly through an edge, pyroomacoustics' general-room search
accepts it from both wall orderings. Its ShoeBox, the analytic image lattice, counts it once.
"""

import numpy as np
import pyroomacoustics as pra

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.room_geometry import RoomGeometry
from compas_acoustics.room_geometry import pra_octave_grid

FS = 48000
DIMS = (10.0, 8.0, 4.0)
ALPHA = [0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25]


def images(source, receiver, order):
    scene = Scene("t")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(*DIMS, materials=AcousticMaterial("wall", ALPHA, 0.1)))
    geometry = RoomGeometry.from_scene(scene)
    room = geometry.to_pra(FS, order)
    room.add_source(list(source))
    room.add_microphone(np.reshape(receiver, (3, 1)))
    room.image_source_model()
    spec = Pair(geometry, np.asarray(source), np.asarray(receiver), np.eye(3), float(FS), air_absorption=False)
    return image_sources(room, 0, 0, spec)


def shoebox(source, receiver, order):
    with pra_octave_grid():
        box = pra.ShoeBox(list(DIMS), fs=FS, max_order=order, use_rand_ism=False, materials=pra.Material(
            {"description": "wall", "coeffs": ALPHA, "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000]}))
        box.add_source(list(source))
        box.add_microphone(list(receiver))
        box.image_source_model()
    return box.sources[0]


def test_an_image_reached_through_an_edge_is_counted_once():
    """Source and receiver at one height and one width, as on a seating grid: paths graze edges."""
    source, receiver = (2.0, 4.0, 1.5), (8.0, 4.0, 1.5)
    ours, box = images(source, receiver, 4), shoebox(source, receiver, 4)
    np.testing.assert_array_equal(np.bincount(ours.order), np.bincount(box.orders))
    distance = np.sort(ours.delay * 343.0)
    expected = np.sort(np.linalg.norm(box.images.T - receiver, axis=1))
    np.testing.assert_allclose(distance, expected, rtol=0, atol=1e-4)


def test_each_kept_image_has_the_shoebox_gains():
    source, receiver = (2.0, 4.0, 1.5), (8.0, 4.0, 1.5)
    ours, box = images(source, receiver, 3), shoebox(source, receiver, 3)
    distance = ours.delay * 343.0
    position = receiver + ours.doa.cartesian / np.linalg.norm(ours.doa.cartesian, axis=1, keepdims=True) * distance[:, None]
    match = [int(np.argmin(np.linalg.norm(box.images.T - p, axis=1))) for p in position]
    assert len(set(match)) == len(match)
    reflection = ours.gain * 4.0 * np.pi * distance[:, None]
    # 125 Hz to 8 kHz, where both rooms take the material's own coefficients
    np.testing.assert_allclose(reflection[:, 1:8], np.asarray(box.damping).T[match][:, 1:8], rtol=1e-6)


def test_distinct_images_are_all_kept():
    ours = images((2.3417, 1.871, 1.433), (7.12, 5.07, 1.61), 3)
    np.testing.assert_array_equal(np.bincount(ours.order), [1, 6, 18, 38])
