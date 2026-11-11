"""Lecture room HIL D 24.1 (ETH Hönggerberg) from a Rhino OBJ export.

The OBJ (Y-up) has one group per layer of ``Acoustics::<material>``. Every surface layer gets the
library material of its name (or of its alias below), the ``sources`` and ``receivers`` layers hold
points that become the scene's sources and receivers, and the furniture and wall linings are
free-standing obstacles inside the closed concrete / plaster / ceiling box.

Every source-receiver pair is simulated twice:

- ``Mono``: image sources and ray tracing, which is also the reference of
- ``Ambisonics``: the hybrid engine with edge diffraction (UDFA) and surface scattering, whose decay
  is fitted to the mono reference. Its W channel is analysed.

Tables go to the console, CSVs and figures (plan, 3D views, parameters, decay curves) to
``results/hil_d24_1``.
"""

import os
import warnings

import numpy as np
from obj_scene_utils import Y_UP_TO_Z_UP
from obj_scene_utils import layer_name
from obj_scene_utils import materials_by_layer
from obj_scene_utils import points_on_layer
from obj_scene_utils import read_obj_point_groups
from obj_scene_utils import set_scattering_by_layer
from results_plots import plot_3d
from results_plots import plot_band_parameters
from results_plots import plot_decay_curves
from results_plots import plot_plan

import compas_acoustics as ca

HERE = os.path.dirname(os.path.abspath(__file__))
OBJ_PATH = os.path.join(HERE, "hil_d24_1.obj")
RESULTS_DIR = os.path.join(HERE, "results", "hil_d24_1")
OBJ_Y_UP = True

SOURCE_LAYER = "sources"
RECEIVER_LAYER = "receivers"

# Layers not named after a library material (compas_acoustics/materials_library)
LAYER_MATERIAL_ALIASES = {
    "plywood": "wood_16mm",
    "ceiling": "ceiling_plasterboard",
    "carpet": "carpet_hairy",
    "plastic_chairs": "chairs_unoccupied_plastic",
    "foam_chairs": "chairs_medium_upholstered",
}
# Scattering coefficient for every band, replacing the library's (most library materials have none: 0.05)
LAYER_SCATTERING = {
    "plywood": 0.3,
    "ceiling": 0.4,
    "plastic_chairs": 0.4,
    "foam_chairs": 0.4,
}
# Layers inside the room rather than part of its closed boundary
OBSTACLE_LAYERS = {"plywood", "carpet", "plastic_chairs", "foam_chairs"}

IMAGE_SOURCE_ORDER = 2
MONO = dict(image_source_order=IMAGE_SOURCE_ORDER, ray_tracing=True, n_rays=30_000, air_absorption=True)
HYBRID = dict(image_source_order=IMAGE_SOURCE_ORDER, diffraction=True, scattering=True, sh_order=1)
SUMMARY_PARAMETERS = ["T30", "EDT", "C80", "D50", "Ts", "STI"]


def load_room(path):
    """The surface groups of the OBJ in Z-up, with their library materials, scattering and roles."""
    objects = ca.AcousticObject.from_obj(path, name="hil_d24_1")
    if OBJ_Y_UP:
        for acoustic_object in objects:
            acoustic_object.acoustic_geometry.transform(Y_UP_TO_Z_UP)
    groups = [o.group for o in objects]
    materials = set_scattering_by_layer(materials_by_layer(groups, LAYER_MATERIAL_ALIASES), LAYER_SCATTERING)
    ca.AcousticObject.assign_materials_by_group(objects, materials)
    ca.AcousticObject.assign_roles_by_group(objects, {g: "obstacle" for g in groups if layer_name(g) in OBSTACLE_LAYERS})
    return objects


def load_sources_and_receivers(path):
    """One source per point on the sources layer and one receiver per point on the receivers layer, in file order."""
    point_groups = read_obj_point_groups(path, y_up=OBJ_Y_UP)
    # compas_acoustics requires lowercase names
    sources = [ca.Source(f"s{i}", *p) for i, p in enumerate(points_on_layer(point_groups, SOURCE_LAYER), 1)]
    receivers = [ca.Receiver(f"r{i}", *p) for i, p in enumerate(points_on_layer(point_groups, RECEIVER_LAYER), 1)]
    if not sources or not receivers:
        raise ValueError(f"No points on the '{SOURCE_LAYER}' or '{RECEIVER_LAYER}' layer of {path}")
    return sources, receivers


def by_pair(rirs):
    """The responses keyed ``'<source>-<receiver>'``, sorted."""
    return {f"{ir.source.name}-{ir.receiver.name}": ir for ir in sorted(rirs.values(), key=lambda ir: (ir.source.name, ir.receiver.name))}


def analyse(rirs, out_dir):
    """:meth:`Analyser.analyse` of every pair (channel 0, W for ambisonics), each also written to ``out_dir/<pair>.csv``."""
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for pair, ir in rirs.items():
        with warnings.catch_warnings():
            # an ambisonic response is analysed on W, its channel 0
            warnings.filterwarnings("ignore", message=".*analysing channel 0 only")
            analyser = ca.Analyser(ir)
            results[pair] = analyser.analyse()
        analyser.export_csv(os.path.join(out_dir, f"{pair}.csv"))
    return results


def print_comparison(results_by_method):
    """The overall (500 / 1000 Hz) values of every pair, one column block per method."""
    methods = list(results_by_method)
    pairs = list(results_by_method[methods[0]])
    print(f"\n{'':8}" + "".join(f"{m:^{9 * len(SUMMARY_PARAMETERS)}}" for m in methods))
    print(f"{'pair':8}" + "".join(f"{p:>9}" for p in SUMMARY_PARAMETERS) * len(methods))
    print("-" * (8 + 9 * len(SUMMARY_PARAMETERS) * len(methods)))
    for pair in pairs:
        row = "".join(f"{results_by_method[m][pair][f'{p}_overall']:>9.2f}" for m in methods for p in SUMMARY_PARAMETERS)
        print(f"{pair:8}{row}")
    for method in methods:
        spans = []
        for p in ("T30", "EDT"):
            v = np.array([r[f"{p}_overall"] for r in results_by_method[method].values()], dtype=float)
            spans.append(f"{p} mean {np.nanmean(v):.2f} s ({np.nanmin(v):.2f} - {np.nanmax(v):.2f})")
        print(f"{method:11} " + ", ".join(spans))


def print_band_tables(results, parameters=("T30", "EDT", "C80", "D50")):
    """Per-band values of every pair."""
    for pair, result in results.items():
        freqs = result["freqs"]
        print(f"\n== {pair}\n{'':6}" + "".join(f"{f:>8}" for f in freqs) + f"{'Overall':>9}")
        for p in parameters:
            print(f"{p:6}" + "".join(f"{v:>8.2f}" for v in result[p]) + f"{result[f'{p}_overall']:>9.2f}")


def main():
    objects = load_room(OBJ_PATH)
    sources, receivers = load_sources_and_receivers(OBJ_PATH)

    scene = ca.Scene(name="hil_d24_1", description="ETH Hönggerberg lecture room HIL D 24.1, from a Rhino OBJ export.")
    scene.add_acoustic_objects(objects)
    for source in sources:
        scene.add_source(source)
    for receiver in receivers:
        scene.add_receiver(receiver)

    print("Surfaces:")
    for o in objects:
        scattering = o.material.scattering_coefficients
        s = f"s={scattering[0]:.2f}" if scattering and len(set(scattering)) == 1 else ("s=per band" if scattering else "s=0.05 (default)")
        print(f"  {o.group:28} -> {o.material.name:28} {o.role:9} {s:17} {o.acoustic_geometry.number_of_faces():4} faces")
    print("Sources:   " + ", ".join(f"{s.name} ({s.x:.2f}, {s.y:.2f}, {s.z:.2f})" for s in sources))
    print("Receivers: " + ", ".join(f"{r.name} ({r.x:.2f}, {r.y:.2f}, {r.z:.2f})" for r in receivers))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_plan(objects, sources, receivers, os.path.join(RESULTS_DIR, "plan.png"))
    plot_3d(objects, sources, receivers, os.path.join(RESULTS_DIR, "view_3d.png"))

    print(f"\nMono: {MONO}")
    mono = by_pair(scene.simulate(ca.Mono(**MONO)))
    print(f"Ambisonics (reference: mono): {HYBRID}")
    hybrid = by_pair(scene.simulate(ca.Ambisonics(reference="mono", **HYBRID)))

    results = {
        "mono": analyse(mono, os.path.join(RESULTS_DIR, "mono")),
        "diffraction": analyse(hybrid, os.path.join(RESULTS_DIR, "diffraction")),
    }
    print("\nPer band, hybrid with diffraction (W channel):")
    print_band_tables(results["diffraction"])
    print_comparison(results)

    for method, rirs in (("mono", mono), ("diffraction", hybrid)):
        plot_band_parameters(results[method], os.path.join(RESULTS_DIR, f"parameters_{method}.png"))
        plot_decay_curves(rirs, os.path.join(RESULTS_DIR, f"decay_{method}.png"))
    print(f"\nFigures and CSVs in {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
