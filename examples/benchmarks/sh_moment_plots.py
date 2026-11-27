"""Explanatory figures for the SH-moment late field: the rooms, and where the late energy arrives from.

For a discussion of :mod:`compas_acoustics.hybrid.moments` against the directional FDN. Per room of
``sh_moment_late_field.py`` (one source, its first receiver, one octave band) it draws:

1. ``<case>_geometry.png``: the shoebox, its faces coloured by absorption, the source, the receivers and the
   early reflections' directions of arrival at the receiver (arrow length and colour: level).
2. ``<case>_doa.png``: maps of azimuth and elevation (room frame) over a few time windows of the late field:

   - the rays' energy per direction of the 480-direction moment grid, the image sources' arrivals overlaid;
   - the same rays as an order-N signal can see them: the steered power ``y(u)^T C y(u)`` of their covariance
     ``C = G M`` (the moments truncated at order 2N);
   - the steered power of the synthesized late field, ``late_field="moments"``;
   - the steered power of the directional FDN's late field, ``late_field="fdn"``.

   Each map is normalised to its own maximum, in dB; the three lower rows are directly comparable.
3. ``<case>_dipoles.png``: the dipole energies along x, y and z relative to their mean over time (1 in a diffuse
   field), rays against both late fields.
4. ``grids.png``: the moment grid (480 directions) against the FDN's 96 lines.

Both late fields use ``late_envelope="directional"`` and seed 42, so their reference, image sources and rays are
the same. Figures go to ``examples/results/sh_moments/plots/``::

    python examples/benchmarks/sh_moment_plots.py
    python examples/benchmarks/sh_moment_plots.py --cases corridor --order 1 --band 500
"""

import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sh_moment_late_field as bench  # noqa: E402

import compas_acoustics as ca  # noqa: E402
import compas_acoustics.simulation_methods.hybrid as hybrid_methods  # noqa: E402
from compas_acoustics.hybrid.envelope import octave_band_split  # noqa: E402
from compas_acoustics.hybrid.moments import covariance  # noqa: E402
from compas_acoustics.hybrid.moments import histogram_moments  # noqa: E402
from compas_acoustics.hybrid.moments import moment_grid  # noqa: E402
from compas_acoustics.hybrid.moments import sh_basis  # noqa: E402
from compas_acoustics.room.geometry import RoomGeometry  # noqa: E402
from compas_acoustics.room.pra import source_to_pra  # noqa: E402
from compas_acoustics.room.raytracing import default_duration  # noqa: E402
from compas_acoustics.room.raytracing import directional_histograms  # noqa: E402

OUT = os.path.join(bench.OUT, "plots")
FS = bench.FS
SEED = bench.SEED
DYNAMIC_DB = 20.0
EARLY_ORDER = 2  # the image sources drawn in the geometry figure
ORDER_COLOURS = ["black", "tab:red", "tab:blue"]
# the late windows, seconds after the last image source's arrival
WINDOWS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.4)]
AZ = np.radians(np.linspace(-180.0, 180.0, 121))
EL = np.radians(np.linspace(-90.0, 90.0, 61))


def unit_grid():
    """Directions (n_el, n_az, 3) of the map, azimuth from +x towards +y, elevation from the horizontal plane."""
    az, el = np.meshgrid(AZ, EL)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def azimuth_elevation(unit):
    unit = np.asarray(unit, dtype=float).reshape(-1, 3)
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    return np.degrees(np.arctan2(unit[:, 1], unit[:, 0])), np.degrees(np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)))


def steered_power(C, rotation, sh_order):
    """``y(u)^T C y(u)`` on the map, ``C`` (K, K) in the receiver's frame, the map in the room's."""
    unit = unit_grid().reshape(-1, 3) @ np.asarray(rotation, dtype=float)
    Y = sh_basis(sh_order, unit)
    return np.einsum("ui,ij,uj->u", Y, C, Y).reshape(len(EL), len(AZ))


def output_covariance(late, band, window):
    """The channel covariance of a late field (n, K) in one octave band over a window of samples."""
    start, stop = window
    x = octave_band_split(np.asarray(late, dtype=float)[start:stop], FS)[band]
    return x.T @ x


def trace_rays(scene, method, receiver, duration):
    """The rays of the methods' reference sorted on the moment grid (room frame): its DirectionalHistogram."""
    geometry = RoomGeometry.from_scene(scene)
    mono = ca.Mono(ray_tracing=True)
    (histogram,) = directional_histograms(
        geometry,
        [source_to_pra(source) for source in scene.sources],
        np.array([receiver.position], dtype=float).T,
        FS,
        [moment_grid() @ receiver.rotation.T],
        max_order=method.image_source_order,
        duration=max(default_duration(geometry, True), duration or 0.0),
        n_rays=method.histogram_rays,
        receiver_radius=mono.receiver_radius,
        energy_threshold=mono.energy_thres,
        seed=SEED,
    )[0]
    return histogram


def simulate(case, sh_order):
    """Both late fields of the first receiver: {"moments": HybridResult, "fdn": HybridResult}, and the scene."""
    scene = bench.scene_of(case)
    results = {}
    method = None
    for late_field in ("moments", "fdn"):
        method = ca.Ambisonics(
            sh_order=sh_order,
            normalization="N3D",
            seed=SEED,
            duration=bench.CASES[case].get("duration"),
            late_field=late_field,
            late_envelope="directional",
        )
        irs = scene.simulate(method)
        results[late_field] = next(method.results[key] for key, ir in irs.items() if ir.receiver.name == "receiver_0")
    method.histogram_rays = int(next(iter(method.reference_rirs.values())).generation["n_rays"])
    return scene, method, results


# ------------------------------------------------------------------------------------------------------- figures


def plot_geometry(case, scene, result, band_hz, path):
    geometry = RoomGeometry.from_scene(scene)
    spec = bench.CASES[case]
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(projection="3d")
    absorption = np.array([float(np.mean(geometry.materials[m].absorption)) for m in geometry.face_materials])
    cmap = plt.get_cmap("viridis")
    polygons = [geometry.vertices[list(face)] for face in geometry.faces]
    ax.add_collection3d(Poly3DCollection(polygons, facecolors=cmap(absorption), alpha=0.22, edgecolors="0.3", linewidths=0.6))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, label="mean absorption")

    source = np.asarray(spec["source"])
    ax.scatter(*source, s=80, c="tab:red", marker="*", label="source", depthshade=False)
    for i, position in enumerate(spec["receivers"]):
        ax.scatter(*position, s=50, c="k", marker="o", depthshade=False)
        ax.text(*position, f"  r{i}", fontsize=9)

    # the early arrivals at receiver_0, from the image sources in the receiver's frame
    images = result.core.images
    receiver = np.asarray(spec["receivers"][0])
    rotation = result.core.pair.receiver_rotation
    b = int(np.argmin(np.abs(np.asarray(images.bands) - band_hz)))
    energy = np.asarray(images.gain[:, b], dtype=float) ** 2
    level = 10 * np.log10(energy / energy.max() + 1e-30)
    keep = (level > -DYNAMIC_DB) & (np.asarray(images.order) <= EARLY_ORDER)
    doa = np.asarray(images.doa.cartesian, dtype=float)[keep]
    doa = doa / np.linalg.norm(doa, axis=1, keepdims=True) @ np.asarray(rotation).T
    length = 0.5 * min(spec["dims"]) * (1.0 + level[keep] / DYNAMIC_DB) + 0.3
    colours = [ORDER_COLOURS[o] for o in np.asarray(images.order)[keep]]
    for d, n, c in zip(doa, length, colours):
        # from where the reflection arrives, pointing at the receiver
        ax.quiver(*(receiver + d * n), *(-d * n), color=c, arrow_length_ratio=0.25, linewidth=1.4)
    for order in range(EARLY_ORDER + 1):
        ax.plot([], [], color=ORDER_COLOURS[order], label=f"arrival at r0, image order {order}")

    ax.set_box_aspect(spec["dims"])
    ax.locator_params(axis="y", nbins=3)
    ax.locator_params(axis="z", nbins=3)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", xlim=(0, spec["dims"][0]), ylim=(0, spec["dims"][1]), zlim=(0, spec["dims"][2]))
    ax.set_title(
        f"{case}: {spec['dims'][0]:g} x {spec['dims'][1]:g} x {spec['dims'][2]:g} m; early arrivals at r0 up to order {EARLY_ORDER}, "
        f"top {DYNAMIC_DB:.0f} dB at {band_hz:g} Hz (length: level)"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _map(ax, values, title):
    db = 10 * np.log10(np.maximum(values, 1e-30) / np.max(values))
    mesh = ax.pcolormesh(np.degrees(AZ), np.degrees(EL), db, vmin=-DYNAMIC_DB / 2, vmax=0, cmap="inferno", shading="auto")
    ax.set_title(f"{title}, spread {-db.min():.1f} dB", fontsize=9)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-90, -45, 0, 45, 90])
    return mesh


def plot_doa(case, histogram, results, sh_order, band, band_hz, path):
    moments_result, fdn_result = results["moments"], results["fdn"]
    rotation = moments_result.core.pair.receiver_rotation
    start = float(np.max(moments_result.core.images.delay))
    hist_band = int(np.argmin(np.abs(np.asarray(histogram.bands) - band_hz)))
    moments = histogram_moments(histogram, 2 * sh_order, rotation)  # (n_bands, n_bins, n_moments)
    az_g, el_g = azimuth_elevation(histogram.directions)
    images = moments_result.core.images
    b = int(np.argmin(np.abs(np.asarray(images.bands) - band_hz)))
    image_energy = np.asarray(images.gain[:, b], dtype=float) ** 2
    image_doa = np.asarray(images.doa.cartesian, dtype=float) @ np.asarray(rotation).T
    az_i, el_i = azimuth_elevation(image_doa)

    rows = ["rays per grid direction", f"rays at order {sh_order}: C = G M", "late_field='moments'", "late_field='fdn'"]
    fig, axes = plt.subplots(len(rows), len(WINDOWS), figsize=(4.2 * len(WINDOWS), 2.4 * len(rows)), sharex=True, sharey=True, squeeze=False)
    for c, (a, z) in enumerate(WINDOWS):
        t0, t1 = start + a, start + z
        first, last = int(t0 / histogram.bin_size), max(int(np.ceil(t1 / histogram.bin_size)), int(t0 / histogram.bin_size) + 1)
        window = (int(t0 * FS), int(t1 * FS))
        label = f"{t0 * 1e3:.0f}-{t1 * 1e3:.0f} ms"

        energy = histogram.energy[:, hist_band, first:last].sum(axis=1)
        ax = axes[0, c]
        hit = energy > 0.0
        level = 10 * np.log10(energy[hit] / max(energy.max(), 1e-30))
        ax.scatter(az_g[~hit], el_g[~hit], s=4, c="0.85")
        ax.scatter(az_g[hit], el_g[hit], c=level, s=18, cmap="inferno", vmin=-DYNAMIC_DB, vmax=0)
        if c == 0:
            # the early arrivals' directions, for orientation: the late field's first window follows them
            keep = image_energy > 10 ** (-DYNAMIC_DB / 10) * image_energy.max()
            ax.scatter(az_i[keep], el_i[keep], s=40, facecolors="none", edgecolors="cyan", linewidths=0.9, label="image sources (earlier)")
            ax.legend(loc="lower left", fontsize=7)
        ax.set_title(f"{label}: {int(hit.sum())} directions hit ({DYNAMIC_DB:.0f} dB scale)", fontsize=9)

        C_rays = covariance(moments[hist_band, first:last].sum(axis=0), sh_order)
        _map(axes[1, c], steered_power(C_rays, rotation, sh_order), label)
        _map(axes[2, c], steered_power(output_covariance(moments_result.late, band, window), rotation, sh_order), label)
        mesh = _map(axes[3, c], steered_power(output_covariance(fdn_result.late, band, window), rotation, sh_order), label)
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row + "\nelevation (deg)", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("azimuth (deg, room frame)")
    fig.subplots_adjust(right=0.88, hspace=0.35)
    fig.colorbar(mesh, cax=fig.add_axes([0.9, 0.12, 0.012, 0.55]), label="steered power y(u)^T C y(u), dB re each map's maximum")
    fig.suptitle(f"{case}, receiver 0, {band_hz:g} Hz octave, order {sh_order}: where the late energy arrives from", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dipoles(case, histogram, results, sh_order, band, band_hz, path, step=0.05):
    """Dipole energies (x, y, z) / their mean over time, windows of ``step`` seconds."""
    rotation = results["moments"].core.pair.receiver_rotation
    start = float(np.max(results["moments"].core.images.delay))
    hist_band = int(np.argmin(np.abs(np.asarray(histogram.bands) - band_hz)))
    moments = histogram_moments(histogram, 2 * sh_order, rotation)
    n = results["moments"].late.shape[0]
    edges = np.arange(start, min(n / FS, histogram.energy.shape[2] * histogram.bin_size), step)
    acn = [3, 1, 2]  # x, y, z

    def dipoles(C):
        d = np.array([C[k, k] for k in acn])
        return d / d.mean() if d.mean() > 0.0 else np.full(3, np.nan)

    series = {"rays (G M)": [], "moments": [], "fdn": []}
    for t0, t1 in zip(edges[:-1], edges[1:]):
        first, last = int(t0 / histogram.bin_size), int(np.ceil(t1 / histogram.bin_size))
        series["rays (G M)"].append(dipoles(covariance(moments[hist_band, first:last].sum(axis=0), sh_order)))
        for key in ("moments", "fdn"):
            series[key].append(dipoles(output_covariance(results[key].late, band, (int(t0 * FS), int(t1 * FS)))))
    centres = 0.5 * (edges[:-1] + edges[1:]) * 1e3
    # the last bin the rays reach in this band; the moments' shape is held from their hold point on
    rays_end = (np.flatnonzero(histogram.energy[:, hist_band].sum(axis=0) > 0.0)[-1] + 1) * histogram.bin_size * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    styles = {"rays (G M)": dict(color="k", lw=2.0), "moments": dict(color="tab:blue", lw=1.2), "fdn": dict(color="tab:orange", lw=1.2, ls="--")}
    for i, axis in enumerate("xyz"):
        for key, values in series.items():
            axes[i].plot(centres, np.asarray(values)[:, i], label=key, **styles[key])
        axes[i].axhline(1.0, color="0.6", lw=0.8)
        axes[i].axvline(rays_end, color="0.4", lw=0.8, ls=":")
        axes[i].set(title=f"dipole {axis}", xlabel="time (ms)")
    axes[0].set_ylabel("energy / mean of x, y, z")
    axes[0].plot([], [], color="0.4", lw=0.8, ls=":", label="last ray")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{case}, receiver 0, {band_hz:g} Hz octave, order {sh_order}: late field anisotropy over time ({step * 1e3:.0f} ms windows)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grids(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (unit, title) in zip(
        axes,
        [
            (moment_grid(), f"moment grid: {len(moment_grid())} directions (the rays' bins)"),
            (hybrid_methods.fdn_directions(96).cartesian, "FDN: 96 delay lines (one direction each)"),
        ],
    ):
        az, el = azimuth_elevation(unit)
        ax.scatter(az, el, s=8, c="k")
        ax.set(title=title, xlabel="azimuth (deg)", xlim=(-180, 180), ylim=(-90, 90))
    axes[0].set_ylabel("elevation (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", default=",".join(bench.CASES))
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--band", type=float, default=1000.0, help="octave band centre in Hz")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(args.out, exist_ok=True)
    band = int(np.argmin(np.abs(np.asarray(bench.BANDS) - args.band)))
    band_hz = bench.BANDS[band]
    plot_grids(os.path.join(args.out, "grids.png"))
    for case in args.cases.split(","):
        print(f"{case}: simulating both late fields at order {args.order}")
        scene, method, results = simulate(case, args.order)
        receiver = next(r for r in scene.receivers if r.name == "receiver_0")
        histogram = trace_rays(scene, method, receiver, bench.CASES[case].get("duration"))
        plot_geometry(case, scene, results["moments"], band_hz, os.path.join(args.out, f"{case}_geometry.png"))
        plot_doa(case, histogram, results, args.order, band, band_hz, os.path.join(args.out, f"{case}_doa.png"))
        plot_dipoles(case, histogram, results, args.order, band, band_hz, os.path.join(args.out, f"{case}_dipoles.png"))
        print(f"{case}: figures in {args.out}")


if __name__ == "__main__":
    main()
