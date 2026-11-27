"""Sources, directions and points as pyroomacoustics objects."""

import math
from typing import Optional
from typing import Union

import numpy as np
import pyroomacoustics as pra
from compas.geometry import Point
from compas.geometry import Vector
from compas.tolerance import TOL

from compas_acoustics.defaults import Defaults
from compas_acoustics.source import Source

__all__ = ["vector_to_pra", "directivity_to_pra", "points_to_array", "source_to_pra"]


def vector_to_pra(vec: Vector) -> pra.DirectionVector:
    """A direction as azimuth and colatitude in radians."""
    if vec.length <= TOL.absolute:
        raise ValueError("Vector length cannot be zero.")
    x, y, z = vec.unitized()
    return pra.DirectionVector(azimuth=math.atan2(y, x), colatitude=math.acos(z), degrees=False)


#: The p of each first-order pattern p + (1 - p) cos(theta)
CARDIOID_P = {
    "bidirectional": 0.0,
    "hypercardioid": 0.25,
    "supercardioid": 0.37,
    "cardioid": 0.5,
    "omnidirectional": 1.0,
}


def directivity_to_pra(dirt, vec: Vector, fs: Optional[int] = None):
    """A first-order pattern of :data:`CARDIOID_P` facing ``vec``, gain 1 on axis; or a measured one (a SOFA path or a
    pyroomacoustics directivity, :func:`compas_acoustics.directivity.to_pra`) at ``fs``, default the package's rate."""
    if isinstance(dirt, str) and dirt in CARDIOID_P:
        return pra.CardioidFamily(orientation=vector_to_pra(vec), p=CARDIOID_P[dirt], gain=1.0)
    from compas_acoustics.directivity import to_pra

    if fs is None:
        fs = Defaults().SAMPLING_RATE
    return to_pra(dirt, list(vec), fs)


def points_to_array(
    pts: Union[Point, list[Point], list[float], list[list[float]]],
) -> np.ndarray:
    """One point or a list of points as a (3, N) array."""
    if isinstance(pts, Point) or (isinstance(pts, list) and isinstance(pts[0], (int, float))):
        pts = [pts]
    return np.array([[*pt] for pt in pts]).T


def source_to_pra(src: Source, fs: Optional[int] = None) -> pra.SoundSource:
    """A source at its position with its directivity (at ``fs`` when measured), and no signal."""
    return pra.SoundSource(
        position=points_to_array(src.position),
        # without a directivity object an omnidirectional source traces the default fibonacci rays
        directivity=None if isinstance(src.directivity, str) and src.directivity == "omnidirectional" else directivity_to_pra(src.directivity, src.direction, fs),
    )
