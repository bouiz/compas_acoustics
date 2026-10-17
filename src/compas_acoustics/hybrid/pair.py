"""One source-receiver pair of a scene, as the ported modules see it.

Stands in for hybrid-rir's ``RoomSpec`` with the same attribute names, so the
ported function bodies, which keep the parameter name ``spec``, read as
upstream's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compas_acoustics.room_geometry import RoomGeometry


@dataclass(frozen=True, eq=False)
class Pair:
    geometry: RoomGeometry
    source: np.ndarray
    receiver: np.ndarray
    receiver_rotation: np.ndarray
    fs: float
    speed_of_sound: float = 343.0
    air_absorption: bool = True

    def __post_init__(self) -> None:
        for name in ("source", "receiver"):
            value = np.asarray(getattr(self, name), dtype=float).ravel()
            if value.shape != (3,):
                raise ValueError(f"{name} must be a length-3 position in metres")
            object.__setattr__(self, name, value)
        rot = np.asarray(self.receiver_rotation, dtype=float)
        if (
            rot.shape != (3, 3)
            or not np.allclose(rot @ rot.T, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(rot), 1.0, atol=1e-8)
        ):
            raise ValueError("receiver_rotation must be a 3x3 proper rotation (orthogonal, determinant +1)")
        object.__setattr__(self, "receiver_rotation", rot)
        if not self.geometry.is_closed:
            raise ValueError("a hybrid response needs a closed room")
