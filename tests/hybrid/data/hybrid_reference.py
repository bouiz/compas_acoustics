"""Write hybrid_reference_82833d1.npz: a short Hybrid response from the code before surface scattering.

Run against a checkout of 82833d1 (``git archive 82833d1``), not this tree:
    PYTHONPATH=<archive>/src python tests/hybrid/data/hybrid_reference.py tests/hybrid/data/hybrid_reference_82833d1.npz
``tests/hybrid/test_scattering.py`` checks that ``Hybrid(scattering=False)`` still gives it.
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


def reference_scene():
    scene = Scene("reference", sampling_rate=32000)
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=AcousticMaterial("wall", 0.1, 0.25)))
    scene.add_source(Source("src", 1.5, 1.5, 1.5))
    scene.add_receiver(Receiver("rec", 4.5, 3.5, 1.2, direction=[0, 1, 0]))
    return scene


def reference_rir(**kwargs):
    hybrid = Hybrid(**SETTINGS, **kwargs)
    reference_scene().simulate(hybrid)
    (result,) = hybrid.results.values()
    return result.rir


if __name__ == "__main__":
    np.savez_compressed(sys.argv[1], rir=reference_rir())
