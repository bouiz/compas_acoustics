import math
from typing import Union, Optional

import numpy as np
import pyroomacoustics as pra
import scipy.io.wavfile as wav
from compas.datastructures import Mesh
from compas.geometry import Point
from compas.geometry import Vector

from compas_acoustics import Defaults
from compas_acoustics.acoustic_material import AcousticMaterial
from compas_acoustics.geometry import AcousticObject
from compas_acoustics.receiver import Receiver
from compas_acoustics.scene import Scene
from compas_acoustics.source import Source


def vector_to_pra(vec: Vector, tol=1e-9) -> pra.DirectionVector:
    """
    Convert a COMPAS Vector to a pyroomacoustics DirectionVector 
    using spherical coordinates (radians).
    """
    if vec.length <= tol:
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
    sound_file = src.sound_file
    fs, audio_clean = wav.read(sound_file)

    if len(audio_clean.shape) > 1:
        audio_clean = audio_clean[:, 0]
    audio_clean = audio_clean.astype(np.float32) / np.max(np.abs(audio_clean))

    return pra.SoundSource(
        position=points_to_array(src.position),
        signal=audio_clean,
        directivity=directivity_to_pra(src.directivity, src.direction),
    )

def receivers_to_pra_mic_array(
    receivers: Union[Receiver, list[Receiver]],
    sample_rate: Optional[int] = None,
) -> pra.MicrophoneArray:
    """
    Convert compas_acoustics Receiver(s) to a pyroomacoustics MicrophoneArray,
    which is a (3, N) 2D numpy array.
    """
    if sample_rate is None:
        sample_rate = Defaults.SAMPLING_RATE

    if isinstance(receivers, Receiver):
        receivers = [receivers]
    positions = [r.position for r in receivers]

    return pra.MicrophoneArray(
        R=points_to_array(positions),
        fs=sample_rate,
        directivity=None  # TODO: Implement this
    )

def material_to_pra(mat: AcousticMaterial) -> pra.Material:
    """Convert a compas_acoustics Material to a pyroomacoustics Material."""
    return pra.Material(
        energy_absorption={
            "description": mat.description,
            "coeffs": mat.absorption_coefficients,
            "center_freqs": mat.frequency_bands
        },
        scattering={
            "description": mat.description,
            "coeffs": mat.scattering_coefficients,
            "center_freqs": mat.frequency_bands
        },
    )

def mesh_to_pra_wall(mesh: Mesh) -> pra.Wall:
    """Convert a triangular compas Mesh to a pyroomacoustics Wall."""
    if len(mesh.vertices) != 3:
        raise ValueError("Mesh must be a triangle.")
    coords = [mesh.geometry.vertex_coordinates(v) for v in mesh.geometry.vertices()]
    corners = points_to_array(coords)
    mat = material_to_pra(mesh.material)

    return pra.Wall(
        corners=corners,
        absorption=mat.absorption_coeffs,
        scattering=mat.scattering_coeffs,
        name = mesh.name
    )

def acoustic_object_to_pra_walls(acoustic_object: AcousticObject) -> list[pra.Wall]:
    """Convert a compas_acoustics AcousticObject to a list of pyroomacoustics Walls."""
    mesh = acoustic_object.acoustic_geometry
    mat = material_to_pra(acoustic_object.material)

    walls = []
    for fkey in mesh.faces():
        # vertices of the face
        face_vertices = mesh.face_vertices(fkey)
        pts = points_to_array([mesh.vertex_coordinates(vkey) for vkey in face_vertices])
            
        # create wall
        wall = pra.wall_factory(pts,
                                mat.absorption_coeffs,
                                mat.scattering_coeffs,
                                name=f"{acoustic_object.name}_{fkey}")
        walls.append(wall)
    return walls

def scene_to_pra_room(scene: Scene, fs: int, max_order: int, ray_tracing: bool, air_absorption: bool) -> pra.Room:
    """Convert a compas_acoustics Scene to a pyroomacoustics Room."""
    # create walls (recurse into groups to reach leaf acoustic objects)
    walls = []
    for obj in scene.acoustic_objects:
        for leaf in obj.iter_acoustic_objects():
            walls.extend(acoustic_object_to_pra_walls(leaf))

    # create the room
    room = pra.Room(
        walls,
        fs=fs,
        max_order=max_order,
        ray_tracing=ray_tracing,
        air_absorption=air_absorption,
    )
    return room