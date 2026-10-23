"""Write short responses of the hybrid engine from an earlier revision of the code, for regression tests.

Run against a checkout of that revision (``git archive <revision>``), or this tree before a refactor:
    PYTHONPATH=<archive>/src python tests/hybrid/data/hybrid_reference.py <out.npz> --format ambisonic|binaural [case ...]
Each case is stored under its name; the ambisonic run also stores the ``AMBISONIC_ONLY`` cases, and every run writes
the seconds each case took to ``<out.npz>.timings.json``.

``hybrid_reference_227725f.npz`` holds the first three cases, ambisonic, from the code with energy_thres 1e-7 and the
early part rendered in one pass for both outputs: ``test_scattering.py`` checks ``scattering=False`` against its
``rir``. ``ambisonic_reference_3b9333d.npz`` and ``binaural_reference_3b9333d.npz`` hold every case, from the code
before the hybrid was split into the binaural and ambisonics methods (``Hybrid(output=...)``, through an earlier
version of this script): ``test_hybrid_outputs.py`` checks them.
"""

import json
import sys
import time

import numpy as np
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.simulation_methods import Ambisonics
from compas_acoustics.simulation_methods import Binaural

SETTINGS = dict(tail="eyring", sh_order=1, image_source_order=2, duration=0.04)
WALL = AcousticMaterial("wall", 0.1, 0.25)
# rir: Eyring tail; scattering: Ewert's tap; ray_tracing: a seeded ray-traced reference, so the late envelope is shaped
CASES = {
    "rir": {},
    "scattering": dict(scattering=True),
    "ray_tracing": dict(tail="ray_tracing"),
    "two_slopes": dict(tail="ray_tracing", n_slopes=2),
    "no_directional_absorption": dict(directional_absorption=False),
    "udfa": dict(diffraction="udfa", panel=True),
    "btms": dict(diffraction="btms", panel=True, image_source_order=1),
    "turned": dict(direction=(-1, 0, 0)),
}
AMBISONIC_ONLY = {"order2_n3d": dict(sh_order=2, normalization="N3D"), "order3": dict(sh_order=3)}


def reference_scene(panel=False, direction=(0, 1, 0)):
    scene = Scene("reference", sampling_rate=32000)
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=WALL))
    if panel:
        # 2 m by 2 m, across the direct path from the source to the receiver
        mesh = Mesh.from_vertices_and_faces([[3, 1.5, 0.5], [3, 3.5, 0.5], [3, 3.5, 2.5], [3, 1.5, 2.5]], [[0, 1, 2, 3]])
        scene.add_acoustic_object(AcousticObject(mesh, WALL, name="panel", role="obstacle"))
    scene.add_source(Source("src", 1.5, 1.5, 1.5))
    scene.add_receiver(Receiver("rec", 4.5, 3.5, 1.2, direction=list(direction)))
    return scene


def _simulate(method, kwargs):
    kwargs = dict(kwargs)
    scene = reference_scene(kwargs.pop("panel", False), kwargs.pop("direction", (0, 1, 0)))
    settings = dict(SETTINGS, **kwargs)
    if method is Binaural:
        settings.pop("sh_order")
    (ir,) = scene.simulate(method(**settings)).values()
    return ir.denormalized()


def reference_rir(**kwargs):
    """The raw ambisonic response of a case, (n, (sh_order + 1) ** 2)."""
    return _simulate(Ambisonics, kwargs)


def reference_binaural(**kwargs):
    """The raw binaural response of a case, (n, 2)."""
    return _simulate(Binaural, kwargs)


if __name__ == "__main__":
    out, args = sys.argv[1], sys.argv[2:]
    output = "ambisonic"
    if "--format" in args:
        i = args.index("--format")
        output = args[i + 1]
        args = args[:i] + args[i + 2:]
    render = reference_rir if output == "ambisonic" else reference_binaural
    cases = {**CASES, **AMBISONIC_ONLY} if output == "ambisonic" else dict(CASES)
    names = args or list(cases)
    responses, timings = {}, {}
    for name in names:
        start = time.perf_counter()
        responses[name] = render(**cases[name])
        timings[name] = time.perf_counter() - start
        print(f"{name}: {timings[name]:.1f} s", flush=True)
    np.savez_compressed(out, **responses)
    with open(f"{out}.timings.json", "w") as f:
        json.dump(timings, f, indent=1)
