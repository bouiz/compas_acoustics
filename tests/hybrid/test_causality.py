"""Nothing is heard before it arrives, through ``Scene.simulate(Ambisonics(...))``.

compas-owned (not ported). The DSP is pinned by the ported test_early_causality.py;
this runs the same L-shaped room through the scene, the room geometry and the method.
"""

import warnings

import numpy as np
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import Ambisonics
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source

FS = 48000
SOURCE, RECEIVER = (7.0, 1.0, 1.5), (4.0, 5.5, 1.5)


def l_room():
    outline = [(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)]
    n = len(outline)
    vertices = [[x, y, 0.0] for x, y in outline] + [[x, y, 3.0] for x, y in outline]
    faces = [list(range(n))[::-1], list(range(n, 2 * n))] + [[k, (k + 1) % n, (k + 1) % n + n, k + n] for k in range(n)]
    scene = Scene("l_room")
    scene.add_acoustic_object(AcousticObject(Mesh.from_vertices_and_faces(vertices, faces), AcousticMaterial("wall", 0.2, 0.1)))
    scene.add_source(Source("src", *SOURCE))
    scene.add_receiver(Receiver("rec", *RECEIVER))
    return scene


@pytest.mark.parametrize("model", ["udfa", "btms"])
def test_diffraction_adds_nothing_before_the_direct_sound_could_arrive(model):
    # Around the corner the wall hides the direct sound; no diffracted path is shorter than it.
    # Before the causal taper: -18 dB (UDFA) and -17 dB (BTMS) of the diffracted energy. What remains is
    # the zero-phase band magnitude and UDFA transfer of each path, which ring before its arrival; how the
    # nine paths' pre-rings add depends on their fractional delays: -38 to -57 dB for speeds of sound
    # 341.6-344.5 m/s (measured). The bound keeps 17 dB below the untapered response.
    scene = l_room()
    responses = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for diffraction in (False, model):
            (responses[diffraction],) = scene.simulate(Ambisonics(tail="eyring", duration=0.5, image_source_order=2,
                                                              diffraction=diffraction, seed=1, normalize=False)).values()
    diffracted = responses[model].ir[:, 0] - responses[False].ir[:, 0]
    hidden = int(np.linalg.norm(np.subtract(SOURCE, RECEIVER)) / responses[model].generation["speed_of_sound"] * FS)
    early = 10 * np.log10(np.sum(diffracted[: hidden - 48] ** 2) / np.sum(diffracted ** 2))
    assert early < -35.0
