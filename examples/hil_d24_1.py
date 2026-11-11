"""Lecture room HIL D 24.1 (ETH Hönggerberg) from a Rhino OBJ export.

The OBJ (Y-up) has one group per layer of ``Acoustics::<material>``. Every surface layer gets the
library material of its name (or of its alias below), the ``sources`` and ``receivers`` layers hold
points that become the scene's sources and receivers, and the furniture and wall linings are
free-standing obstacles inside the closed concrete / plaster / ceiling box. Every source-receiver pair
is simulated with image sources and ray tracing, analysed, and written to ``results/hil_d24_1``.
"""

import os

import numpy as np
from obj_scene_utils import Y_UP_TO_Z_UP
from obj_scene_utils import layer_name
from obj_scene_utils import materials_by_layer
from obj_scene_utils import points_on_layer
from obj_scene_utils import read_obj_point_groups
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
    "plastic_chairs": "chairs_unoccupied_plastic",
    "foam_chairs": "chairs_medium_upholstered",
}
# Layers inside the room rather than part of its closed boundary
OBSTACLE_LAYERS = {"plywood", "carpet", "plastic_chairs", "foam_chairs"}

SIMULATION = dict(image_source_order=3, ray_tracing=True, n_rays=20_000, air_absorption=True)


def load_room(path):
    """The surface groups of the OBJ in Z-up, with their library materials and roles."""
    objects = ca.AcousticObject.from_obj(path, name="hil_d24_1")
    if OBJ_Y_UP:
        for acoustic_object in objects:
            acoustic_object.acoustic_geometry.transform(Y_UP_TO_Z_UP)
    groups = [o.group for o in objects]
    ca.AcousticObject.assign_materials_by_group(objects, materials_by_layer(groups, LAYER_MATERIAL_ALIASES))
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


def print_summary(results):
    """The overall (500 / 1000 Hz) values of every pair."""
    params = ["T30", "EDT", "C80", "D50", "Ts", "STI", "G"]
    print(f"\n{'pair':8}" + "".join(f"{p:>9}" for p in params))
    print("-" * (8 + 9 * len(params)))
    for pair, result in results.items():
        print(f"{pair:8}" + "".join(f"{result[f'{p}_overall']:>9.2f}" for p in params))
    t30 = np.array([r["T30_overall"] for r in results.values()], dtype=float)
    print(f"\nT30 (500/1000 Hz) over all pairs: mean {np.nanmean(t30):.2f} s, range {np.nanmin(t30):.2f} - {np.nanmax(t30):.2f} s")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    objects = load_room(OBJ_PATH)
    sources, receivers = load_sources_and_receivers(OBJ_PATH)

    scene = ca.Scene(name="HIL D 24.1", description="ETH Hönggerberg lecture room, from a Rhino OBJ export.")
    scene.add_acoustic_objects(objects)
    for source in sources:
        scene.add_source(source)
    for receiver in receivers:
        scene.add_receiver(receiver)

    print("Surfaces:")
    for o in objects:
        print(f"  {o.group:28} -> {o.material.name:28} {o.role:9} {o.acoustic_geometry.number_of_faces():4} faces")
    print("Sources:  " + ", ".join(f"{s.name} ({s.x:.2f}, {s.y:.2f}, {s.z:.2f})" for s in sources))
    print("Receivers: " + ", ".join(f"{r.name} ({r.x:.2f}, {r.y:.2f}, {r.z:.2f})" for r in receivers))

    rirs = scene.simulate(ca.Mono(**SIMULATION))
    by_pair = {f"{ir.source.name}-{ir.receiver.name}": ir for ir in sorted(rirs.values(), key=lambda ir: (ir.source.name, ir.receiver.name))}

    results = {}
    for pair, ir in by_pair.items():
        analyser = ca.Analyser(ir)
        results[pair] = analyser.analyse()
        print(f"\n== {pair}")
        analyser.print_table(["T30", "EDT", "C80", "D50"], show_overall=True)
        analyser.export_csv(os.path.join(RESULTS_DIR, f"{pair}.csv"))

    print_summary(results)
    plot_plan(objects, sources, receivers, os.path.join(RESULTS_DIR, "plan.png"))
    plot_band_parameters(results, os.path.join(RESULTS_DIR, "parameters.png"))
    plot_decay_curves(by_pair, os.path.join(RESULTS_DIR, "decay.png"))
    print(f"\nFigures and CSVs in {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
