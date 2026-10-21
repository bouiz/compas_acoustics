"""The air a room is filled with: temperature, relative humidity and static pressure.

The speed of sound after Cramer (1993) and the air absorption after ISO 9613-1:1993,
both from :mod:`pyfar.constants` (``speed_of_sound_cramer``, ``air_attenuation``).
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["AirConditions", "air_or_default", "speed_of_sound"]


@dataclass(frozen=True)
class AirConditions:
    """Air at a temperature (deg C), relative humidity (%) and static pressure (Pa).

    The ranges are those of Cramer's speed of sound as pyfar implements it: 0 to
    30 deg C and 75 to 102 kPa, with any relative humidity from 0 to 100 %.
    """

    temperature: float = 20.0
    humidity: float = 50.0
    pressure: float = 101325.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 30.0:
            raise ValueError(f"temperature must be within 0 to 30 deg C (Cramer), got {self.temperature!r}")
        if not 0.0 <= self.humidity <= 100.0:
            raise ValueError(f"humidity must be a relative humidity within 0 to 100 %, got {self.humidity!r}")
        if not 75e3 <= self.pressure <= 102e3:
            raise ValueError(f"pressure must be within 75 to 102 kPa (Cramer), in Pa, got {self.pressure!r}")

    @property
    def speed_of_sound(self) -> float:
        """Speed of sound in m/s, after Cramer (1993)."""
        try:
            from pyfar.constants import speed_of_sound_cramer
        except ImportError:  # pyfar < 0.8, on Python 3.9 (Rhino 8)
            from compas_acoustics.room._pyfar_constants import speed_of_sound_cramer

        return float(speed_of_sound_cramer(self.temperature, self.humidity / 100.0, atmospheric_pressure=self.pressure))

    def air_attenuation(self, frequencies) -> np.ndarray:
        """Energy attenuation coefficient ``m`` per metre at ``frequencies`` in Hz (above 50 Hz), after ISO 9613-1."""
        try:
            from pyfar.constants import air_attenuation
        except ImportError:
            from compas_acoustics.room._pyfar_constants import air_attenuation

        frequencies = np.asarray(frequencies, dtype=float)
        _, m, _ = air_attenuation(self.temperature, frequencies, self.humidity / 100.0,
                                  atmospheric_pressure=self.pressure)
        return np.asarray(m.freq, dtype=float).reshape(frequencies.shape)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional[AirConditions]:
        return None if not data else cls(**data)


def air_or_default(air: Optional[AirConditions]) -> AirConditions:
    """The air, or ``AirConditions()`` (20 deg C, 50 %, 101325 Pa), which every simulation uses without one."""
    return AirConditions() if air is None else air


def speed_of_sound(air: Optional[AirConditions]) -> float:
    """The air's speed of sound, the default air's without one."""
    return air_or_default(air).speed_of_sound
