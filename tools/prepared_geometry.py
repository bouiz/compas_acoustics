"""Record and compare what ``prepare_mesh`` makes of a fixed set of scenes, face by face.

A simulation of a prepared mesh is a deterministic function of the prepared mesh, so a change to the preparation that
leaves every scene's prepared output identical leaves every simulation of it identical. This tool checks that, and
lists what changed where it is not identical.

``collect <corpus>`` records the inputs of every ``prepare_mesh`` call (vertices, faces, objects, materials, seeds,
options, forced two-sided faces) made by ``examples/benchmarks/hil_d24_1.py``'s room (as ``RoomGeometry.from_scene(scene,
prepare=PrepareOptions())`` prepares it) and by the test files given (default :data:`TEST_FILES`), one scene per call.
:func:`capturing` records the calls of any other caller (an application) the same way.

``dump <corpus> <out>`` prepares every scene of a corpus with the code imported and stores the per-face output:
normals, winding flips, two-sidedness, face classes, air and other hits, components and whether they close, walls,
enclosing objects, seed statuses, the air volume, the key stats and the orientation conflicts.

``compare <a> <b>`` reports, per scene, whether the two dumps are identical, and otherwise which faces changed and
under which orientation rule they fall.

    python tools/prepared_geometry.py collect examples/results/geometry/corpus.pkl
    python tools/prepared_geometry.py dump examples/results/geometry/corpus.pkl examples/results/geometry/<commit>.pkl
    python tools/prepared_geometry.py compare examples/results/geometry/<a>.pkl examples/results/geometry/<b>.pkl
"""

import argparse
import contextlib
import hashlib
import os
import pickle
import subprocess
import sys
from dataclasses import asdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples", "benchmarks")

#: Test files whose prepare_mesh calls make the corpus (missing ones are skipped).
TEST_FILES = [
    "tests/test_geometry_prepare.py",
    "tests/test_geometry_report.py",
    "tests/test_geometry_roles.py",
    "tests/test_open_rooms.py",
    "tests/test_prepared_accounting.py",
    "tests/test_room_from_prepared.py",
    "tests/test_sector_edges.py",
    "tests/test_sector_wiring.py",
    "tests/test_closed_solid_orientation.py",
]

STATS = [
    "welded_faces",
    "components",
    "closed_components",
    "boundary_edges",
    "non_manifold_edges",
    "flipped_faces",
    "two_sided_faces",
    "skipped_faces",
    "simulated_walls",
    "rays_per_seed",
    "leak_fraction",
    "outdoor",
]


@contextlib.contextmanager
def capturing(scenes, label=lambda: "call", modules=()):
    """Record the inputs of every ``prepare_mesh`` call made inside the block into the list ``scenes``.

    ``label()`` names each call (a test id, a scene name); identical inputs are recorded once. ``modules`` are other
    modules that imported ``prepare_mesh`` by name (an application's), patched too.
    """
    import compas_acoustics.geometry as geometry
    import compas_acoustics.geometry.prepare as prepare

    original = prepare.prepare_mesh
    patched = [geometry, prepare, *modules]
    seen = set()

    def recording(vertices, faces, object_face_ranges, face_materials, face_scattering=None, seeds=(), options=None, force_two_sided=None):
        args = {
            "vertices": np.asarray(vertices, dtype=np.float64).reshape(-1, 3).copy(),
            "faces": np.asarray(faces, dtype=np.int64).reshape(-1, 3).copy(),
            "object_face_ranges": {str(k): [int(v[0]), int(v[1])] for k, v in object_face_ranges.items()},
            "face_materials": {int(k): v for k, v in face_materials.items()},
            "face_scattering": None if face_scattering is None else {int(k): float(v) for k, v in face_scattering.items()},
            "seeds": np.asarray(seeds, dtype=np.float64).reshape(-1, 3).copy(),
            "options": None if options is None else asdict(options),
            "force_two_sided": None if force_two_sided is None else np.asarray(force_two_sided, dtype=bool).copy(),
        }
        key = hashlib.sha1(pickle.dumps(args)).hexdigest()
        if key not in seen:
            seen.add(key)
            name = label()
            names = {s["name"] for s in scenes}
            k, unique = 1, name
            while unique in names:
                k += 1
                unique = f"{name} #{k}"
            scenes.append({"name": unique, "args": args})
        return original(vertices, faces, object_face_ranges, face_materials, face_scattering, seeds, options, force_two_sided)

    for module in patched:
        module.prepare_mesh = recording
    try:
        yield scenes
    finally:
        for module in patched:
            module.prepare_mesh = original


def _hil_scene():
    sys.path.insert(0, EXAMPLES)
    import hil_d24_1 as example

    import compas_acoustics as ca

    scene = ca.Scene(name="hil_d24_1")
    scene.add_acoustic_objects(example.load_room(example.OBJ_PATH))
    sources, receivers = example.load_sources_and_receivers(example.OBJ_PATH)
    for source in sources:
        scene.add_source(source)
    for receiver in receivers:
        scene.add_receiver(receiver)
    return scene


def collect(corpus, test_files):
    import pytest

    import compas_acoustics as ca
    from compas_acoustics.geometry import PrepareOptions

    scenes = []
    with capturing(scenes, lambda: "hil_d24_1"):
        ca.RoomGeometry.from_scene(_hil_scene(), prepare=PrepareOptions())

    current = {"id": "?"}

    class Label:
        def pytest_runtest_setup(self, item):
            current["id"] = item.nodeid

    files = [f for f in test_files if os.path.exists(os.path.join(ROOT, f))]
    with capturing(scenes, lambda: current["id"]):
        os.chdir(ROOT)
        pytest.main([*files, "-q", "-p", "no:cacheprovider", "-p", "no:randomly"], plugins=[Label()])
    os.makedirs(os.path.dirname(os.path.abspath(corpus)), exist_ok=True)
    with open(corpus, "wb") as f:
        pickle.dump(scenes, f)
    print(f"{len(scenes)} scenes from hil_d24_1 and {len(files)} test files -> {corpus}")


def _run(args):
    from compas_acoustics.geometry import PrepareOptions
    from compas_acoustics.geometry import prepare_mesh

    options = None if args["options"] is None else PrepareOptions(**args["options"])
    return prepare_mesh(
        args["vertices"],
        args["faces"],
        args["object_face_ranges"],
        args["face_materials"],
        args["face_scattering"],
        args["seeds"],
        options,
        force_two_sided=args["force_two_sided"],
    )


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def dump(corpus, out):
    import compas_acoustics

    with open(corpus, "rb") as f:
        scenes = pickle.load(f)
    result = {"_meta": {"compas_acoustics": compas_acoustics.__file__, "commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", "src"))}}
    for scene in scenes:
        m = _run(scene["args"])
        vis = m.visibility
        result[scene["name"]] = {
            "objects": list(m.objects),
            "face_object": np.asarray(m.face_object),
            "faces": np.asarray(m.faces),
            "normals": np.asarray(m.normals),
            "flipped": np.asarray(m.flipped),
            "two_sided": np.asarray(m.two_sided),
            "face_class": np.asarray(m.face_class),
            "air_hits": np.asarray(m.air_hits),
            "other_hits": np.asarray(m.other_hits),
            "component": np.asarray(m.component),
            "component_closed": np.asarray(m.component_closed),
            "walls": [tuple(int(f) for f in w.faces) for w in m.walls],
            "enclosing_objects": [int(o) for o in m.enclosing_objects],
            "seed_status": [s.status for s in vis.seeds],
            "air_volume": (float(m.air_volume), m.air_volume_method),
            "stats": {k: m.stats[k] for k in STATS if k in m.stats},
            "conflicts": m.stats.get("orientation_conflicts", []),
        }
    with open(out, "wb") as f:
        pickle.dump(result, f)
    print(f"{len(scenes)} scenes prepared with {compas_acoustics.__file__} -> {out}")


def _rule(a, f):
    """The orientation rule face ``f`` of dump ``a`` falls under."""
    c = a["component"][f]
    if a["component_closed"][c]:
        return "closed component"
    if a["face_object"][f] in a["enclosing_objects"]:
        return "enclosing loose faces"
    return "open/non-manifold"


def compare(path_a, path_b):
    with open(path_a, "rb") as f:
        dump_a = pickle.load(f)
    with open(path_b, "rb") as f:
        dump_b = pickle.load(f)
    meta_a, meta_b = dump_a.pop("_meta"), dump_b.pop("_meta")
    print(f"a: {meta_a['commit'][:8]}{' (dirty)' if meta_a['dirty'] else ''} {meta_a['compas_acoustics']}")
    print(f"b: {meta_b['commit'][:8]}{' (dirty)' if meta_b['dirty'] else ''} {meta_b['compas_acoustics']}")
    identical = 0
    for name in dump_a:
        a, b = dump_a[name], dump_b.get(name)
        if b is None:
            print(f"\n[{name}] missing in b")
            continue
        if not np.array_equal(a["faces"].shape, b["faces"].shape) or a["objects"] != b["objects"]:
            print(f"\n[{name}] different faces or objects: not comparable")
            continue
        winding = np.flatnonzero(np.any(a["faces"] != b["faces"], axis=1))
        two_sided = np.flatnonzero(a["two_sided"] != b["two_sided"])
        face_class = np.flatnonzero(a["face_class"] != b["face_class"])
        hits = np.flatnonzero((a["air_hits"] != b["air_hits"]) | (a["other_hits"] != b["other_hits"]))
        # stats the older code does not record (new keys) are shown with a change, not counted as one
        common = set(a["stats"]) & set(b["stats"])
        stats_a, stats_b = a["stats"], b["stats"]
        other = {key: (a[key], b[key]) for key in ("enclosing_objects", "seed_status", "air_volume", "conflicts") if a[key] != b[key]}
        if any(stats_a[k] != stats_b[k] for k in common):
            other["stats"] = (stats_a, stats_b)
        walls = a["walls"] != b["walls"]
        if not (len(winding) or len(two_sided) or len(face_class) or len(hits) or other or walls):
            identical += 1
            continue
        extra = ", ".join(f"{k} {stats_b[k]}" for k in sorted(set(stats_b) - set(stats_a)))
        print(f"\n[{name}] {len(a['faces'])} faces, seeds {','.join(a['seed_status'])}" + (f" (b: {extra})" if extra else ""))
        for label, ids in (("winding", winding), ("two-sided", two_sided), ("class", face_class), ("hits", hits)):
            if not len(ids):
                continue
            groups = {}
            for f in ids:
                obj = a["objects"][a["face_object"][f]] if a["face_object"][f] >= 0 else "-"
                groups.setdefault((obj, _rule(a, f)), []).append(int(f))
            parts = [f"{obj} [{rule}]: {len(fs)}" for (obj, rule), fs in groups.items()]
            print(f"  {label} changed on {len(ids)} face(s): " + "; ".join(parts))
        if walls:
            print(f"  walls: {len(a['walls'])} -> {len(b['walls'])} (merge changed)")
        for key, (va, vb) in other.items():
            if key == "stats":
                diff = {k: (va.get(k), vb.get(k)) for k in set(va) | set(vb) if va.get(k) != vb.get(k)}
                print("  stats: " + ", ".join(f"{k} {x} -> {y}" for k, (x, y) in sorted(diff.items())))
            else:
                print(f"  {key}: {va} -> {vb}")
    print(f"\n{identical}/{len(dump_a)} scenes identical")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    col = sub.add_parser("collect")
    col.add_argument("corpus")
    col.add_argument("tests", nargs="*", default=TEST_FILES)
    dmp = sub.add_parser("dump")
    dmp.add_argument("corpus")
    dmp.add_argument("out")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.corpus, args.tests)
    elif args.command == "dump":
        dump(args.corpus, args.out)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
