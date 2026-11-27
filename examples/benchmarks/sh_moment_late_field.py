"""The ambisonic late field from spherical harmonic energy moments against the directional FDN.

Three shoeboxes of increasing anisotropy (a diffuse office; a room with an absorbing floor and ceiling, whose late
field turns horizontal; a corridor with absorbing side walls, whose late field runs along its axis), two receivers
each, simulated with ``Ambisonics`` at first and third order in four late-field settings: ``late_field`` "moments" or
"fdn", ``late_envelope`` "omni" or "directional". Every run traces its own reference with the method's seed (42), so
the references, the image sources and the rays are the same in all of them.

Per pair it records, against the reference (``Mono``) and against the rays:

- the omnidirectional room parameters of the W channel (T30, EDT, C50, C80, D50) and their differences from the
  reference's, and the late field's energy per octave band against the reference's late part (reference minus the
  hybrid's early field);
- the late field's spatial statistics over a late window (from the last image source to the reference's -30 dB
  point), per octave band 250 Hz to 4 kHz: the normalised channel covariance ``R = C / C_00`` against the rays'
  ``G M / M_0`` (the moments of the same rays sorted on the moment grid), its relative Frobenius error, the
  diffuseness ``psi = 1 - |C_0,xyz| / (sqrt(3) C_00)``, and the dipole energies along x, y and z relative to their
  mean (1 in a diffuse field);
- the time spent preparing and rendering the late field (the network's design and run, or the moments' synthesis),
  per pair, and the whole simulation.

Results go to ``examples/results/sh_moments/`` (``results.json``, ``summary.md``)::

    python examples/benchmarks/sh_moment_late_field.py
    python examples/benchmarks/sh_moment_late_field.py --orders 1 --cases diffuse_office
"""

import argparse
import json
import os
import threading
import time
import warnings
from contextlib import contextmanager

import numpy as np

import compas_acoustics as ca
import compas_acoustics.simulation_methods.hybrid as hybrid_methods
from compas_acoustics.hybrid.envelope import octave_band_split
from compas_acoustics.hybrid.moments import covariance
from compas_acoustics.hybrid.moments import histogram_moments
from compas_acoustics.hybrid.moments import moment_grid
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.raytracing import default_duration
from compas_acoustics.room.raytracing import directional_histograms

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "examples", "results", "sh_moments")
FS = 48000
SEED = 42
BANDS = [62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
SPATIAL_BANDS = slice(2, 7)  # 250 Hz to 4 kHz
SETTINGS = {
    "moments/omni": dict(late_field="moments"),
    "moments/directional": dict(late_field="moments", late_envelope="directional"),
    "fdn/omni": dict(late_field="fdn"),
    "fdn/directional": dict(late_field="fdn", late_envelope="directional"),
}


def material(name, absorption, scattering=0.1):
    return ca.AcousticMaterial(name, absorption, scattering)


CASES = {
    "diffuse_office": dict(
        dims=(6.0, 5.0, 3.0),
        materials=material("wall", 0.2),
        source=(1.5, 1.5, 1.6),
        receivers=[(4.0, 3.0, 1.2), (5.2, 0.8, 1.2)],
    ),
    "absorbing_floor_ceiling": dict(
        dims=(8.0, 6.0, 3.0),
        materials={
            "floor": material("carpet", 0.6, 0.05),
            "ceiling": material("absorber", 0.8, 0.05),
            "left_wall": material("plaster", 0.03, 0.05),
            "right_wall": material("plaster", 0.03, 0.05),
            "front_wall": material("plaster", 0.03, 0.05),
            "back_wall": material("plaster", 0.03, 0.05),
        },
        source=(2.0, 2.0, 1.5),
        receivers=[(5.5, 4.0, 1.2), (7.0, 1.0, 1.6)],
        # Eyring predicts 0.5 s; the horizontal energy between the plaster walls decays at about 1 s, and a reference
        # cut at 0.5 s (28 to 35 dB down) would end the decay its T30 is measured on
        duration=2.0,
    ),
    "corridor": dict(
        dims=(24.0, 2.5, 3.0),
        materials={
            "left_wall": material("absorber", 0.5, 0.05),
            "right_wall": material("absorber", 0.5, 0.05),
            "floor": material("carpet", 0.3, 0.05),
            "ceiling": material("absorber", 0.5, 0.05),
            "front_wall": material("concrete", 0.02, 0.05),
            "back_wall": material("concrete", 0.02, 0.05),
        },
        source=(3.0, 1.2, 1.5),
        receivers=[(12.0, 1.3, 1.5), (20.0, 0.8, 1.2)],
        # Eyring predicts 0.5 s here; the axial energy between the end walls decays over more than 1 s, and a
        # reference cut at 0.5 s would end the decay it is compared with
        duration=2.5,
    ),
}


class Timer:
    """Wall time of the late field's preparation (``prepare``) and rendering (``render_ambisonics``), summed over pairs."""

    def __init__(self):
        self.lock = threading.Lock()
        self.prepare = 0.0
        self.render = 0.0

    @contextmanager
    def installed(self):
        prepare, render = hybrid_methods.prepare, hybrid_methods.render_ambisonics

        def timed(fn, slot):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    with self.lock:
                        setattr(self, slot, getattr(self, slot) + time.perf_counter() - start)

            return wrapper

        hybrid_methods.prepare = timed(prepare, "prepare")
        hybrid_methods.render_ambisonics = timed(render, "render")
        try:
            yield self
        finally:
            hybrid_methods.prepare, hybrid_methods.render_ambisonics = prepare, render


def scene_of(case):
    spec = CASES[case]
    scene = ca.Scene(case)
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(*spec["dims"], materials=spec["materials"]))
    scene.add_source(ca.Source("source", *spec["source"]))
    for i, position in enumerate(spec["receivers"]):
        scene.add_receiver(ca.Receiver(f"receiver_{i}", *position))
    return scene


def band_energy(x):
    return (octave_band_split(np.asarray(x, dtype=float), FS) ** 2).sum(axis=1)


def w_parameters(ir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = ca.Analyser(ir).analyse()
    out = {}
    for key in ("T30", "EDT"):
        v = np.asarray(values[key], dtype=float)
        out[key] = float(np.nanmean(v[2:5]))  # the analyser's bands start at 125 Hz: 500 Hz to 2 kHz
    for key in ("C50", "C80", "D50"):
        out[key] = float(values.get(f"{key}_overall", np.nanmean(np.asarray(values[key], dtype=float))))
    return out


def late_window(result, reference):
    """From the last image source to the reference's -30 dB point (broadband), in samples."""
    start = int(np.ceil(float(np.max(result.core.images.delay)) * FS)) + 1
    edc = np.cumsum(reference[::-1] ** 2)[::-1]
    stop = int(np.flatnonzero(edc <= 1e-3 * edc[0])[0]) if np.any(edc <= 1e-3 * edc[0]) else reference.size
    return start, max(stop, start + int(0.05 * FS))


def spatial(late, truth_moments, window, bin_size, sh_order):
    """Per band of SPATIAL_BANDS: the covariance error, the diffuseness and the dipole energies, output and rays."""
    start, stop = window
    bands = octave_band_split(np.asarray(late, dtype=float)[start:stop], FS)  # (9, n, K)
    first, last = int(start / (bin_size * FS)), int(np.ceil(stop / (bin_size * FS)))
    rows = []
    for b in range(*SPATIAL_BANDS.indices(9)):
        C = bands[b].T @ bands[b]
        R = C / C[0, 0]
        M = truth_moments[b, first:last].sum(axis=0)
        T = covariance(M, sh_order) / M[0]
        rows.append(
            {
                "band": BANDS[b],
                "covariance_error": float(np.linalg.norm(R - T) / np.linalg.norm(T)),
                "psi": float(1.0 - np.linalg.norm(R[0, 1:4]) / np.sqrt(3.0)),
                "psi_rays": float(1.0 - np.linalg.norm(T[0, 1:4]) / np.sqrt(3.0)),
                # ACN 1, 2, 3 are y, z, x
                "dipoles": [float(R[3, 3]), float(R[1, 1]), float(R[2, 2])],
                "dipoles_rays": [float(T[3, 3]), float(T[1, 1]), float(T[2, 2])],
            }
        )
    return rows


def truth(scene, method, sh_order, duration=None):
    """The rays of the methods' reference, sorted on the moment grid: moments (9, n_bins, (2N+1)^2) per receiver."""
    geometry = RoomGeometry.from_scene(scene)
    mono = ca.Mono(ray_tracing=True)
    receivers = list(scene.receivers)
    directions = [moment_grid() @ receiver.rotation.T for receiver in receivers]
    histograms = directional_histograms(
        geometry,
        [source_to_pra(source) for source in scene.sources],
        np.array([r.position for r in receivers], dtype=float).T,
        FS,
        directions,
        max_order=method.image_source_order,
        duration=max(default_duration(geometry, True), duration or 0.0),
        n_rays=method.histogram_rays,
        receiver_radius=mono.receiver_radius,
        energy_threshold=mono.energy_thres,
        seed=SEED,
    )
    return {receiver.name: (histogram_moments(h[0], 2 * sh_order, receiver.rotation), h[0].bin_size) for receiver, h in zip(receivers, histograms)}


def run_case(case, orders):
    scene = scene_of(case)
    records = []
    for sh_order in orders:
        truths = None
        for label, settings in SETTINGS.items():
            method = ca.Ambisonics(sh_order=sh_order, normalization="N3D", seed=SEED, duration=CASES[case].get("duration"), **settings)
            timer = Timer()
            start = time.perf_counter()
            with timer.installed():
                irs = scene.simulate(method)
            total = time.perf_counter() - start
            # the rays every setting traces its reference with (the late envelope's rule on the 96 lines' grid)
            method.histogram_rays = hybrid_methods._HybridMethod._histogram_rays(RoomGeometry.from_scene(scene).to_pra(FS, 0), hybrid_methods.fdn_directions(96).cartesian)
            if truths is None:
                truths = truth(scene, method, sh_order, CASES[case].get("duration"))
            for key, ir in irs.items():
                result = method.results[key]
                reference_ir = method.reference_rirs[key]
                reference = np.asarray(reference_ir.denormalized(), dtype=float).ravel()
                n = min(reference.size, result.early.shape[0])
                late_target = band_energy(reference[:n]) - band_energy(result.early[:n, 0])
                # a band whose early field holds as much as the reference has no late part to compare with
                valid = late_target > 0.05 * band_energy(reference[:n])
                late_level = np.where(valid, 10 * np.log10(band_energy(result.late[:n, 0]) / np.where(valid, late_target, 1.0)), np.nan)
                params, params_ref = w_parameters(ir), w_parameters(reference_ir)
                moments, bin_size = truths[ir.receiver.name]
                records.append(
                    {
                        "case": case,
                        "sh_order": sh_order,
                        "setting": label,
                        "receiver": ir.receiver.name,
                        "late_energy": ir.generation.get("late_energy", "network (fdn)"),
                        "seconds_total": total,
                        "seconds_prepare": timer.prepare / len(irs),
                        "seconds_render": timer.render / len(irs),
                        "w": params,
                        "w_reference": params_ref,
                        "late_level_db": [None if np.isnan(v) else float(v) for v in late_level],
                        "spatial": spatial(result.late, moments, late_window(result, reference), bin_size, sh_order),
                    }
                )
                print(
                    f"{case:24s} N={sh_order} {label:20s} {ir.receiver.name}: prepare {timer.prepare / len(irs):5.2f} s, render {timer.render / len(irs):5.2f} s; "
                    f"dT30 {params['T30'] - params_ref['T30']:+.3f} s, dC50 {params['C50'] - params_ref['C50']:+.2f} dB; "
                    f"cov err {np.mean([r['covariance_error'] for r in records[-1]['spatial']]):.3f}"
                )
    return records


def summary(records):
    lines = ["# SH-moment late field against the FDN", ""]
    lines.append("Means over the receivers. dX: the W channel's parameter minus the reference's (T30, EDT: mean 500 Hz - 2 kHz).")
    lines.append("Late level: the late field's energy minus the reference's late part (reference minus the hybrid's early field), mean |dB| 125 Hz - 8 kHz.")
    lines.append("Cov err: |R - R_rays| / |R_rays|, R = C / C_00, mean 250 Hz - 4 kHz. psi: diffuseness (rays in brackets).")
    lines.append("psi and dipoles x/y/z (each dipole's energy relative to W's, 1 in a diffuse field): mean 250 Hz - 4 kHz, rays in brackets.")
    lines.append("")
    lines.append("| case | N | setting | prepare s | render s | dT30 s | dEDT s | dC50 dB | dC80 dB | late level dB | cov err | psi (rays) | dipoles x/y/z (rays) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    keys = sorted({(r["case"], r["sh_order"], r["setting"]) for r in records}, key=lambda k: (list(CASES).index(k[0]), k[1], list(SETTINGS).index(k[2])))
    for case, order, setting in keys:
        rows = [r for r in records if (r["case"], r["sh_order"], r["setting"]) == (case, order, setting)]

        def mean(f):
            return float(np.mean([f(r) for r in rows]))

        def over_bands(r, key):
            return np.mean([s[key] for s in r["spatial"]], axis=0)

        dip = np.mean([over_bands(r, "dipoles") for r in rows], axis=0)
        dip_rays = np.mean([over_bands(r, "dipoles_rays") for r in rows], axis=0)
        lines.append(
            f"| {case} | {order} | {setting} | {mean(lambda r: r['seconds_prepare']):.2f} | {mean(lambda r: r['seconds_render']):.2f} "
            f"| {mean(lambda r: r['w']['T30'] - r['w_reference']['T30']):+.3f} | {mean(lambda r: r['w']['EDT'] - r['w_reference']['EDT']):+.3f} "
            f"| {mean(lambda r: r['w']['C50'] - r['w_reference']['C50']):+.2f} | {mean(lambda r: r['w']['C80'] - r['w_reference']['C80']):+.2f} "
            f"| {mean(lambda r: np.nanmean(np.abs(np.array(r['late_level_db'][1:8], dtype=float)))):.2f} "
            f"| {mean(lambda r: np.mean([s['covariance_error'] for s in r['spatial']])):.3f} "
            f"| {mean(lambda r: over_bands(r, 'psi')):.2f} ({mean(lambda r: over_bands(r, 'psi_rays')):.2f}) "
            f"| {dip[0]:.2f}/{dip[1]:.2f}/{dip[2]:.2f} ({dip_rays[0]:.2f}/{dip_rays[1]:.2f}/{dip_rays[2]:.2f}) |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--orders", default="1,3")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(args.out, exist_ok=True)
    records = []
    for case in args.cases.split(","):
        records += run_case(case, [int(o) for o in args.orders.split(",")])
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(records, f, indent=1)
    text = summary(records)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
