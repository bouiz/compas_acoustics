"""One source-receiver pair of a scene, the ``spec`` the hybrid modules take."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from compas_acoustics.room.diffuse import air_attenuation
from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.geometry import RoomGeometry

from .bands import N_BANDS


@dataclass(frozen=True, eq=False)
class Pair:
    """A source and a receiver in a room with a volume (closed, or prepared with its air volume), positions in metres.

    The columns of ``receiver_rotation`` are the receiver's axes in room coordinates; ``air`` None is
    ``AirConditions()``.
    """

    geometry: RoomGeometry
    source: np.ndarray
    receiver: np.ndarray
    receiver_rotation: np.ndarray
    fs: float
    speed_of_sound: float = 343.0
    air_absorption: bool = True
    air: Optional[AirConditions] = None
    #: False for a pair that needs no room volume (the mono method's diffracted paths)
    require_volume: bool = True

    def __post_init__(self) -> None:
        for name in ("source", "receiver"):
            value = np.asarray(getattr(self, name), dtype=float).ravel()
            if value.shape != (3,):
                raise ValueError(f"{name} must be a length-3 position in metres")
            object.__setattr__(self, name, value)
        rot = np.asarray(self.receiver_rotation, dtype=float)
        if rot.shape != (3, 3) or not np.allclose(rot @ rot.T, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(rot), 1.0, atol=1e-8):
            raise ValueError("receiver_rotation must be a 3x3 proper rotation (orthogonal, determinant +1)")
        object.__setattr__(self, "receiver_rotation", rot)
        if self.require_volume and not getattr(self.geometry, "has_volume", self.geometry.is_closed):
            raise ValueError("a hybrid response needs a closed room, or a prepared one with its air volume")

    def air_attenuation(self) -> np.ndarray:
        """Energy attenuation coefficient per metre on the band grid in the pair's air, zero without air absorption."""
        if not self.air_absorption:
            return np.zeros(N_BANDS)
        return air_attenuation(self.air)
