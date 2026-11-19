"""Shoebox rooms, source-receiver pairs and their image sources for the hybrid tests."""

from dataclasses import replace

import numpy as np

from compas_acoustics.hybrid import early
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.room.geometry import RoomGeometry

# x = 0, x = length, y = 0, y = width, z = 0, z = height
WALLS = ("back_wall", "front_wall", "right_wall", "left_wall", "floor", "ceiling")


def box_room(dimensions, walls, floor=None, ceiling=None):
    """A closed shoebox mesh from the origin, one face per wall in ``WALLS`` order; ``walls`` a SurfaceMaterial."""
    lx, ly, lz = (float(d) for d in dimensions)
    v = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0], [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]], dtype=float)
    faces = ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7))
    chosen = dict.fromkeys(WALLS, walls)
    chosen.update({name: m for name, m in (("floor", floor), ("ceiling", ceiling)) if m is not None})
    materials = {name: replace(m, name=name) for name, m in chosen.items()}
    return RoomGeometry.from_mesh(v, faces, WALLS, materials)


def pair(geometry, source, receiver, receiver_rotation=None, fs=48000.0, speed_of_sound=343.0, air_absorption=True):
    """A :class:`Pair` facing +x by default."""
    rotation = np.eye(3) if receiver_rotation is None else receiver_rotation
    return Pair(geometry, source, receiver, rotation, float(fs), speed_of_sound, air_absorption)


def images(spec, max_order):
    """The pair's image sources up to ``max_order``."""
    room = spec.geometry.to_pra(int(spec.fs), max_order)
    room.add_source(np.asarray(spec.source, dtype=float).tolist())
    room.add_microphone(np.asarray(spec.receiver, dtype=float).reshape(3, 1))
    room.image_source_model()
    return early.image_sources(room, 0, 0, spec)
