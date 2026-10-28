"""Write the compas_pb payloads the C# tests read (src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper.Tests/Fixtures).

Run after a schema change, from the repository root, with compas_acoustics installed::

    python tests/pb_fixtures/generate.py

Every guid is fixed, so the payloads name the same objects on both sides.
"""

import base64
import shutil
from pathlib import Path
from uuid import UUID

import numpy as np
from compas_pb import pb_dump_bts

import compas_acoustics.pb  # noqa: F401  (registers the converters)
from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import ImpulseResponse
from compas_acoustics import Ray
from compas_acoustics import RayField
from compas_acoustics import Receiver
from compas_acoustics import ScatteredRay
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.pb.types import Audio

HERE = Path(__file__).parent
CSHARP = HERE.parents[1] / "src" / "compas_acoustics_ghcsharp" / "CompasAcoustics.Grasshopper.Tests" / "Fixtures"


def _guid(n):
    return UUID(int=n)


def _identified(obj, n):
    obj._guid = _guid(n)
    return obj


def source():
    return _identified(Source("src", 1, 2, 3, directivity="cardioid", direction=[0, 1, 0]), 1)


def receiver():
    return _identified(Receiver("rec", 5, 3, 1.5), 2)


def scene():
    s = _identified(Scene("room", sampling_rate=44100), 3)
    objects = AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=_identified(AcousticMaterial("wall", [0.2] * 7, [0.1] * 7), 4))
    for i, obj in enumerate(objects):
        _identified(obj, 10 + i)
    s.add_acoustic_objects(objects)
    s.add_source(_identified(Source("src", 1.5, 1.5, 1.6), 5))
    s.add_receiver(receiver())
    return s


def ir():
    raw = np.array([[1.0, 0.5], [0.5, 0.25], [0.25, 0.125], [0.0, -0.5]]) * 0.1
    return _identified(ImpulseResponse("ir", source(), receiver(), raw, 48000, normalized=True,
                                       generation={"method": "test"}), 6)


def rayfield():
    ray = Ray([[0, 0, 1], [3, 0, 1], [3, 4, 1]], source_index=0, ray_index=0)
    ray.scattered_reflections = [ScatteredRay([[3, 0, 1], [3, 0, 3]], parent_ray_index=0, energy=0.5, origin_reflection_order=1,
                                              source_index=0, ray_index=0, distance_offset=3.0)]
    _identified(ray, 7)
    _identified(ray.scattered_reflections[0], 8)
    ray.source = source()
    ray.scattered_reflections[0].source = ray.source
    return _identified(RayField([ray]), 9)


def audio():
    a = Audio(np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]), 44100, "binaural")
    a.guid = _guid(20)
    return a


FIXTURES = {"scene": scene, "source": source, "ir": ir, "rayfield": rayfield, "audio": audio}


def main():
    CSHARP.mkdir(parents=True, exist_ok=True)
    for name, make in FIXTURES.items():
        path = HERE / f"{name}.b64"
        path.write_text(base64.b64encode(pb_dump_bts(make())).decode() + "\n")
        shutil.copy2(path, CSHARP / path.name)
        print("wrote", path)


if __name__ == "__main__":
    main()
