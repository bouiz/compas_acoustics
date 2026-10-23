"""The room a simulation runs in: its validated geometry and pyroomacoustics build (:mod:`.geometry`),
the air that fills it (:mod:`.air`), its diffuse-field prediction (:mod:`.diffuse`), the conversion
of sources, receivers and scenes to pyroomacoustics (:mod:`.pra`), and pyroomacoustics' runs on it:
image sources, ray tracing and directional histograms (:mod:`.raytracing`)."""

from .air import AirConditions
from .geometry import RoomGeometry

__all__ = ["AirConditions", "RoomGeometry"]
