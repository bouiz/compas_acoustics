import math
from typing import Optional
from typing import Union

import numpy as np
import pyroomacoustics as pra
import scipy.io.wavfile as wav
from compas.geometry import Point
from compas.geometry import Vector
from compas.tolerance import TOL

from compas_acoustics import Defaults
from compas_acoustics.receiver import Receiver
from compas_acoustics.room_geometry import RoomGeometry
from compas_acoustics.scene import Scene
from compas_acoustics.source import Source

__all__ = ["vector_to_pra", "directivity_to_pra", "points_to_array", "source_to_pra", "receivers_to_pra_mic_array", "scene_to_pra_room"]


def vector_to_pra(vec: Vector, tol=None) -> pra.DirectionVector:
    """
    Convert a COMPAS Vector to a pyroomacoustics DirectionVector 
    using spherical coordinates (radians).
    """
    if vec.length <= (TOL.absolute if tol is None else tol):
        raise ValueError("Vector length cannot be zero.")
    
    x, y, z = vec.unitized()

    return pra.DirectionVector(
        azimuth=math.atan2(y, x),
        colatitude=math.acos(z),
        degrees=False  # NOTE: Always use radians?
    )

def directivity_to_pra(dirt: str, vec: Vector) -> pra.CardioidFamily:
    """Convert a Directivity. Only support CardioidFamily now."""
    supported_directivity = {
        "bidirectional": 0.0,
        "hypercardioid": 0.25,
        "supercardioid": 0.37,
        "cardioid": 0.5,
        "omnidirectional": 1.0,
    }

    if dirt not in supported_directivity.keys():
        raise ValueError(f"Directivity '{dirt}' is not supported.")
    
    return pra.CardioidFamily(
        orientation=vector_to_pra(vec),
        p=supported_directivity[dirt],
        gain=1.0
    )

def points_to_array(
    pts: Union[Point, list[Point], list[float], list[list[float]]],
) -> np.ndarray:
    """Convert a single point or a list of points to a (3, N) 2D numpy array."""
    if isinstance(pts, Point) or (isinstance(pts, list) and isinstance(pts[0], (int, float))):
        pts = [pts]
    return np.array([[*pt] for pt in pts]).T

def source_to_pra(src : Source) -> pra.SoundSource:
    """Convert a compas_acoustics Source to a pyroomacoustics SoundSource."""
    if src.sound_file:
        sound_file = src.sound_file
        fs, audio_clean = wav.read(sound_file)

        if len(audio_clean.shape) > 1:
            audio_clean = audio_clean[:, 0]
        audio_clean = audio_clean.astype(np.float32) / np.max(np.abs(audio_clean))
    else:
        audio_clean = None

    return pra.SoundSource(
        position=points_to_array(src.position),
        signal=audio_clean,
        # An omnidirectional source is left without a directivity object, so
        # pyroomacoustics traces its default Fibonacci rays, as hybrid-rir does.
        directivity=None if src.directivity == "omnidirectional" else directivity_to_pra(src.directivity, src.direction),
    )

def receivers_to_pra_mic_array(
    receivers: Union[Receiver, list[Receiver]],
    sampling_rate: Optional[int] = None,
) -> pra.MicrophoneArray:
    """
    Convert compas_acoustics Receiver(s) to a pyroomacoustics MicrophoneArray,
    which is a (3, N) 2D numpy array.
    """
    if sampling_rate is None:
        sampling_rate = Defaults().SAMPLING_RATE

    if isinstance(receivers, Receiver):
        receivers = [receivers]
    positions = [r.position for r in receivers]

    return pra.MicrophoneArray(
        R=points_to_array(positions),
        fs=sampling_rate,
        directivity=None  # TODO: Implement this
    )

def scene_to_pra_room(scene: Scene, fs: int, max_order: int, ray_tracing: bool, air_absorption: bool) -> pra.Room:
    """Convert a compas_acoustics Scene to a pyroomacoustics Room, via :class:`RoomGeometry`."""
    return RoomGeometry.from_scene(scene).to_pra(
        fs, max_order, ray_tracing=ray_tracing, air_absorption=air_absorption
    )
