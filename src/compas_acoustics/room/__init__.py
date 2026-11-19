"""The room a simulation runs in: its geometry (:mod:`.geometry`), air (:mod:`.air`), diffuse-field
prediction (:mod:`.diffuse`), the scene as pyroomacoustics objects (:mod:`.pra`), and the image source,
ray tracing and directional histogram runs (:mod:`.raytracing`)."""

from .air import AirConditions
from .geometry import RoomGeometry

__all__ = ["AirConditions", "RoomGeometry"]
