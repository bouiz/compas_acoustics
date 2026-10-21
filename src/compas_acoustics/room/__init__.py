"""The room a simulation runs in: its validated geometry and pyroomacoustics build (:mod:`.geometry`),
the air that fills it (:mod:`.air`), its diffuse-field prediction (:mod:`.diffuse`), and the conversion
of sources, receivers and scenes to pyroomacoustics (:mod:`.pra`)."""

from .air import AirConditions
from .geometry import RoomGeometry

__all__ = ["AirConditions", "RoomGeometry"]
