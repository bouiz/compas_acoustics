"""Record and compare a regression reference: the HIL D 24.1 example and the test suite.

``record <dir>`` simulates every source-receiver pair of ``examples/hil_d24_1.py`` with the
example's own ``Mono`` and ``Ambisonics`` settings and stores, per method and pair, the raw
response (``ImpulseResponse.denormalized()``), its ``generation`` metadata and the full
``Analyser.analyse()`` output. It then runs the test suite and stores each test's outcome.

``compare <reference dir> <new dir>`` reports what changed between two recordings: responses
(identical, or the largest difference relative to the reference peak), analysis values, test
outcomes and run times.

    python tools/regression_reference.py record examples/results/reference/<commit>
    python tools/regression_reference.py compare examples/results/reference/<a> examples/results/reference/<b>
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")

# Metrics compared with an absolute tolerance (dB for levels and ratios, s for times)
METRIC_KEYS = ["EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts", "STI", "G", "SPL", "LF", "LFC"]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int, bool, np.bool_)):
        return value.item() if hasattr(value, "item") else value
    if value is None or isinstance(value, str):
        return value
    return repr(value)


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _simulate_example(out_dir):
    sys.path.insert(0, EXAMPLES)
    import hil_d24_1 as example

    import compas_acoustics as ca

    objects = example.load_room(example.OBJ_PATH)
    sources, receivers = example.load_sources_and_receivers(example.OBJ_PATH)
    scene = ca.Scene(name="hil_d24_1")
    scene.add_acoustic_objects(objects)
    for source in sources:
        scene.add_source(source)
    for receiver in receivers:
        scene.add_receiver(receiver)

    methods = {
        "mono": lambda: ca.Mono(**example.MONO),
        "diffraction": lambda: ca.Ambisonics(reference="mono", **example.HYBRID),
    }
    timings, summary = {}, {}
    for name, make in methods.items():
        start = time.perf_counter()
        rirs = example.by_pair(scene.simulate(make()))
        timings[name] = time.perf_counter() - start
        arrays, generation, analysis = {}, {}, {}
        for pair, ir in rirs.items():
            arrays[pair] = np.asarray(ir.denormalized(), dtype=np.float64)
            generation[pair] = _jsonable(ir.generation)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*analysing channel 0 only")
                analysis[pair] = _jsonable(ca.Analyser(ir).analyse())
        np.savez_compressed(os.path.join(out_dir, f"{name}_irs.npz"), **arrays)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump({"generation": generation, "analysis": analysis}, f, indent=1)
        summary[name] = {"pairs": len(rirs), "seconds": round(timings[name], 2)}
        print(f"{name}: {len(rirs)} pairs in {timings[name]:.1f} s")
    return summary


def _run_tests(out_dir, workers):
    xml = os.path.join(out_dir, "pytest.xml")
    cmd = [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"]
    if workers:
        cmd += ["-n", str(workers)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    with open(os.path.join(out_dir, "pytest.log"), "w", encoding="utf-8") as f:
        f.write(proc.stdout + "\n" + proc.stderr)
    tail = [line for line in proc.stdout.splitlines() if line.strip()][-1:] or [""]
    print(f"tests: {tail[0]} ({seconds:.0f} s)")
    return {"seconds": round(seconds, 1), "summary": tail[0], "returncode": proc.returncode}


def record(out_dir, skip_tests=False, workers=4):
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", "src", "examples", "requirements.txt")),
        "python": sys.version.split()[0],
        "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import pyroomacoustics as pra

        meta["pyroomacoustics"] = pra.__version__
    except ImportError:
        pass
    meta["example"] = _simulate_example(out_dir)
    if not skip_tests:
        meta["tests"] = _run_tests(out_dir, workers)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def _outcomes(xml_path):
    if not os.path.exists(xml_path):
        return {}
    outcomes = {}
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        name = f"{case.get('classname')}::{case.get('name')}"
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                outcome = child.tag
        outcomes[name] = outcome
    return outcomes


def _metric_diffs(ref, new):
    """Largest absolute difference of every metric over pairs and bands (overall values included)."""
    diffs = {}
    for pair, ref_values in ref.items():
        new_values = new.get(pair, {})
        for key in METRIC_KEYS:
            for suffix in ("", "_overall"):
                a, b = ref_values.get(key + suffix), new_values.get(key + suffix)
                if a is None and b is None:
                    continue
                a = np.atleast_1d(np.array(a if a is not None else np.nan, dtype=float))
                b = np.atleast_1d(np.array(b if b is not None else np.nan, dtype=float))
                if a.shape != b.shape:
                    diffs[key + suffix] = float("inf")
                    continue
                both_nan = np.isnan(a) & np.isnan(b)
                d = np.where(both_nan, 0.0, np.abs(a - b))
                d = np.where(np.isnan(d), np.inf, d)
                diffs[key + suffix] = max(diffs.get(key + suffix, 0.0), float(np.max(d)) if d.size else 0.0)
    return diffs


def compare(ref_dir, new_dir, metric_tol=1e-9):
    with open(os.path.join(ref_dir, "meta.json")) as f:
        ref_meta = json.load(f)
    with open(os.path.join(new_dir, "meta.json")) as f:
        new_meta = json.load(f)
    print(f"reference {ref_meta['commit'][:8]}{' (dirty)' if ref_meta.get('dirty') else ''} vs new {new_meta['commit'][:8]}{' (dirty)' if new_meta.get('dirty') else ''}")
    changed = False

    for method in ref_meta["example"]:
        ref_irs = np.load(os.path.join(ref_dir, f"{method}_irs.npz"))
        new_path = os.path.join(new_dir, f"{method}_irs.npz")
        if not os.path.exists(new_path):
            print(f"\n[{method}] missing in the new recording")
            changed = True
            continue
        new_irs = np.load(new_path)
        identical, worst_db, worst_pair, shape_changes = 0, -np.inf, None, []
        for pair in ref_irs.files:
            a = ref_irs[pair]
            if pair not in new_irs.files:
                shape_changes.append(f"{pair}: missing")
                continue
            b = new_irs[pair]
            if a.shape != b.shape:
                shape_changes.append(f"{pair}: {a.shape} -> {b.shape}")
                n = min(len(a), len(b))
                a, b = a[:n], b[:n]
            if np.array_equal(a, b):
                identical += 1
                continue
            peak = float(np.max(np.abs(a))) or 1.0
            db = 20 * np.log10(max(float(np.max(np.abs(a - b))), 1e-300) / peak)
            if db > worst_db:
                worst_db, worst_pair = db, pair
        n = len(ref_irs.files)
        print(f"\n[{method}] responses: {identical}/{n} bit-identical", end="")
        if worst_pair:
            print(f"; largest difference {worst_db:.1f} dB re peak ({worst_pair})", end="")
            changed = True
        print()
        for line in shape_changes:
            print(f"  length: {line}")
            changed = True

        with open(os.path.join(ref_dir, f"{method}.json")) as f:
            ref_json = json.load(f)
        with open(os.path.join(new_dir, f"{method}.json")) as f:
            new_json = json.load(f)
        diffs = _metric_diffs(ref_json["analysis"], new_json["analysis"])
        moved = {k: v for k, v in diffs.items() if v > metric_tol}
        if moved:
            changed = True
            print("  metrics, largest absolute change: " + ", ".join(f"{k} {v:.3g}" for k, v in sorted(moved.items())))
        else:
            print("  metrics: unchanged")
        gen_changes = sorted(
            {k for p in ref_json["generation"] for k in set(ref_json["generation"][p]) ^ set(new_json["generation"].get(p, {}))}
            | {k for p in ref_json["generation"] for k, v in ref_json["generation"][p].items() if new_json["generation"].get(p, {}).get(k, v) != v}
        )
        if gen_changes:
            print("  generation keys added/removed/changed: " + ", ".join(gen_changes))
        print(f"  time: {ref_meta['example'][method]['seconds']} s -> {new_meta['example'][method]['seconds']} s")

    ref_tests, new_tests = _outcomes(os.path.join(ref_dir, "pytest.xml")), _outcomes(os.path.join(new_dir, "pytest.xml"))
    if ref_tests and new_tests:
        diff = []
        for name in sorted(set(ref_tests) | set(new_tests)):
            a, b = ref_tests.get(name, "absent"), new_tests.get(name, "absent")
            if a != b:
                diff.append(f"  {a:>7} -> {b:<7} {name}")
        regressions = [d for d in diff if "-> failure" in d or "-> error" in d or ("passed ->" in d and "absent" not in d.split("->")[1])]
        print(f"\ntests: {ref_meta['tests']['summary']} -> {new_meta['tests']['summary']} ({ref_meta['tests']['seconds']} s -> {new_meta['tests']['seconds']} s)")
        added = [d for d in diff if d.strip().startswith("absent")]
        if added:
            print(f"  {len(added)} new test(s)")
        for line in [d for d in diff if not d.strip().startswith("absent")]:
            print(line)
        changed = changed or bool(regressions)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("out_dir")
    rec.add_argument("--skip-tests", action="store_true")
    rec.add_argument("--workers", type=int, default=4, help="pytest-xdist workers (0 = serial)")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("reference")
    cmp_.add_argument("new")
    args = parser.parse_args()
    if args.command == "record":
        record(args.out_dir, args.skip_tests, args.workers)
    else:
        sys.exit(1 if compare(args.reference, args.new) else 0)


if __name__ == "__main__":
    main()
