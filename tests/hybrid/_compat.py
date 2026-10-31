"""Compact room types for the hybrid tests, built on this package. Test-only.

``Material``, ``MeshGeometry``, ``ShoeboxGeometry`` and ``RoomSpec`` return a
:class:`SurfaceMaterial`, a :class:`RoomGeometry` and a :class:`Pair`;
``image_sources`` runs the image source model for a ``RoomSpec``.

An open mesh warns instead of raising (RoomGeometry keeps open scenes for ray
tracing), and ``ShoeboxGeometry`` is a closed mesh, so pyroomacoustics builds
it from walls rather than as a ``ShoeboxRoom``.
"""

from dataclasses import replace

import numpy as np

from compas_acoustics.hybrid.bands import BANDS
from compas_acoustics.hybrid.bands import check_bands
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.geometry import SurfaceMaterial


class Material:
    """``Material(absorption, bands, scattering=None)``: per-band coefficients checked, as a :class:`SurfaceMaterial`."""

    def __new__(cls, absorption, bands=None, scattering=None):
        if bands is not None:
            check_bands(bands)
        values = {}
        for kind, value in (("absorption", absorption), ("scattering", scattering)):
            if value is None:
                values[kind] = None
                continue
            value = np.asarray(value, dtype=float)
            if value.shape != BANDS.shape:
                raise ValueError(f"{kind} must have shape {BANDS.shape}, got {value.shape}")
            if np.any(value < 0.0) or np.any(value > 1.0):
                raise ValueError(f"{kind} coefficients must lie in [0, 1]")
            values[kind] = value
        return SurfaceMaterial(name="", absorption=values["absorption"], scattering=values["scattering"])

    @classmethod
    def from_scalar(cls, alpha, scattering=None):
        return cls(np.full(BANDS.size, float(alpha)), BANDS.copy(),
                   None if scattering is None else np.full(BANDS.size, float(scattering)))


def MeshGeometry(vertices, faces, face_materials, materials, obstacles=()):
    named = {name: replace(material, name=name) for name, material in materials.items()}
    return RoomGeometry.from_mesh(vertices, faces, face_materials, named, obstacles=tuple(obstacles))


class ShoeboxGeometry:
    """``ShoeboxGeometry(dimensions, materials)`` as a closed mesh, one face per wall."""

    WALL_NAMES = ("west", "east", "south", "north", "floor", "ceiling")

    def __new__(cls, dimensions, materials):
        if set(materials) != set(cls.WALL_NAMES):
            raise ValueError(f"materials must have exactly the keys {sorted(cls.WALL_NAMES)}")
        lx, ly, lz = (float(d) for d in dimensions)
        v = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
                      [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]], dtype=float)
        faces = {
            "west": (0, 4, 7, 3), "east": (1, 2, 6, 5), "south": (0, 1, 5, 4),
            "north": (3, 7, 6, 2), "floor": (0, 3, 2, 1), "ceiling": (4, 5, 6, 7),
        }
        return MeshGeometry(v, [faces[n] for n in cls.WALL_NAMES], cls.WALL_NAMES, materials)


def RoomSpec(geometry, source, receiver, receiver_rotation=None, fs=48000.0, speed_of_sound=343.0, air_absorption=True):
    rotation = np.eye(3) if receiver_rotation is None else receiver_rotation
    return Pair(geometry, source, receiver, rotation, float(fs), speed_of_sound, air_absorption)


def image_sources(spec, max_order):
    from compas_acoustics.hybrid import early

    room = spec.geometry.to_pra(int(spec.fs), max_order)
    room.add_source(np.asarray(spec.source, dtype=float).tolist())
    room.add_microphone(np.asarray(spec.receiver, dtype=float).reshape(3, 1))
    room.image_source_model()
    return early.image_sources(room, 0, 0, spec)
