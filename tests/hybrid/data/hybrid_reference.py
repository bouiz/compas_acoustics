"""Write short ambisonic Hybrid responses from an earlier revision of the code, for regression tests.

Run against a checkout of that revision (``git archive <revision>``), not this tree:
    PYTHONPATH=<archive>/src python tests/hybrid/data/hybrid_reference.py <out.npz> [case ...]
Each case is stored under its name; the default, ``rir``, is the only one in
``hybrid_reference_82833d1.npz``, which ``tests/hybrid/test_scattering.py`` checks
``Hybrid(scattering=False)`` against. ``hybrid_reference_f658127.npz`` holds every case, the code
before the hybrid rendered ambisonic and binaural outputs from one run (``test_hybrid_outputs.py``).
"""

import sys

import numpy as np

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.simulation_methods import Hybrid

SETTINGS = dict(tail="eyring", sh_order=1, image_source_order=2, duration=0.04)
# rir: Eyring tail; scattering: Task 1's tap; ray_tracing: a seeded ray-traced reference, so the late envelope is shaped
CASES = {"rir": {}, "scattering": dict(scattering=True), "ray_tracing": dict(tail="ray_tracing")}


def reference_scene():
    scene = Scene("reference", sampling_rate=32000)
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=AcousticMaterial("wall", 0.1, 0.25)))
    scene.add_source(Source("src", 1.5, 1.5, 1.5))
    scene.add_receiver(Receiver("rec", 4.5, 3.5, 1.2, direction=[0, 1, 0]))
    return scene


def reference_rir(**kwargs):
    hybrid = Hybrid(**dict(SETTINGS, **kwargs))
    reference_scene().simulate(hybrid)
    (result,) = hybrid.results.values()
    return result.rir


if __name__ == "__main__":
    np.savez_compressed(sys.argv[1], **{case: reference_rir(**CASES[case]) for case in sys.argv[2:] or ["rir"]})
