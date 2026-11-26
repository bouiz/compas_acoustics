"""How far two regression recordings' hybrid responses lie from their own mono run, per room parameter.

``tools/regression_reference.py compare`` says what changed between two recordings; this says which of them is
closer to the reference the hybrid follows: the example's ``Mono`` (ray-traced) responses of the same pairs. For
``foa`` and ``diffraction`` and every parameter, the mean over pairs of |hybrid - mono|, overall value (and the
largest over the per-band values).

    python examples/benchmarks/regression_against_mono.py examples/results/reference/<before> examples/results/reference/<after>
"""

import json
import os
import sys

import numpy as np

PARAMETERS = ["T30", "T20", "EDT", "C50", "C80", "D50", "Ts", "STI", "G"]


def load(directory, method):
    with open(os.path.join(directory, f"{method}.json")) as f:
        return json.load(f)["analysis"]


def distances(directory, method):
    mono, hybrid = load(directory, "mono"), load(directory, method)
    out = {}
    for key in PARAMETERS:
        overall, bands = [], []
        for pair, values in hybrid.items():
            a = values.get(f"{key}_overall", values.get(key))
            b = mono[pair].get(f"{key}_overall", mono[pair].get(key))
            overall.append(abs(float(np.nanmean(np.atleast_1d(np.array(a, dtype=float)))) - float(np.nanmean(np.atleast_1d(np.array(b, dtype=float))))))
            if isinstance(values.get(key), list):
                bands.append(np.nanmax(np.abs(np.array(values[key], dtype=float) - np.array(mono[pair][key], dtype=float))))
        out[key] = (float(np.mean(overall)), float(np.max(bands)) if bands else float("nan"))
    return out


def main(before, after):
    for method in ("foa", "diffraction"):
        print(f"\n[{method}] |hybrid - mono|: overall mean over pairs (largest band)")
        print(f"{'':6s} {'before':>18s} {'after':>18s}")
        a, b = distances(before, method), distances(after, method)
        for key in PARAMETERS:
            print(f"{key:6s} {a[key][0]:8.3f} ({a[key][1]:6.2f}) {b[key][0]:8.3f} ({b[key][1]:6.2f})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
