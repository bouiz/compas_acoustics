"""Figures comparing the analysed responses of several source-receiver pairs, saved to files."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BAND_PARAMETERS = [("T30", "T30 (s)"), ("EDT", "EDT (s)"), ("C80", "C80 (dB)"), ("D50", "D50 (%)")]
DECAY_FLOOR_DB = -80.0
FIGURE_DPI = 150


def plot_band_parameters(results, filepath):
    """T30, EDT, C80 and D50 per octave band, one line per pair; ``results`` maps a pair label to :meth:`Analyser.analyse`."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (param, label) in zip(axes.flat, BAND_PARAMETERS):
        for pair, result in results.items():
            ax.plot([str(f) for f in result["freqs"]], np.asarray(result[param], dtype=float), marker="o", label=pair)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Octave band (Hz)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_decay_curves(rirs, filepath):
    """The broadband impulse response and Schroeder decay of every pair (channel 0: W of an ambisonic response);
    ``rirs`` maps a pair label to an :class:`ImpulseResponse`."""
    fig, (ax_ir, ax_edc) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for pair, ir in rirs.items():
        signal = np.asarray(ir.ir, dtype=float).reshape(len(ir.ir), -1)[:, 0]
        t = np.arange(len(signal)) / ir.sampling_rate
        energy = signal**2
        edc = np.cumsum(energy[::-1])[::-1]
        edc_db = 10 * np.log10(np.maximum(edc / edc[0], 10 ** (DECAY_FLOOR_DB / 10)))
        level_db = 10 * np.log10(np.maximum(energy / energy.max(), 10 ** (DECAY_FLOOR_DB / 10)))
        ax_ir.plot(t, level_db, linewidth=0.5, alpha=0.7, label=pair)
        ax_edc.plot(t, edc_db, label=pair)
    ax_ir.set_ylabel("Level re peak (dB)")
    ax_edc.set_ylabel("Schroeder decay (dB)")
    ax_edc.set_xlabel("Time (s)")
    for ax in (ax_ir, ax_edc):
        ax.set_ylim(DECAY_FLOOR_DB, 3)
        ax.grid(alpha=0.3)
    ax_edc.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIGURE_DPI)
    plt.close(fig)


def _material_colors(objects):
    """One colour per material name, in the order the objects use them."""
    cmap = plt.get_cmap("tab10")
    colors = {}
    for acoustic_object in objects:
        colors.setdefault(acoustic_object.material.name, cmap(len(colors) % 10))
    return colors


def plot_3d(objects, sources, receivers, filepath, views=((30, -60), (25, 120)), hidden_layers=("ceiling",)):
    """Axonometric views of the scene's faces coloured by material, with the sources and receivers.

    Boundary faces are drawn faint so the inside stays visible; groups whose layer is in ``hidden_layers``
    (default the ceiling) are left out. One subplot per ``(elevation, azimuth)`` in ``views``.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = _material_colors(objects)
    shown = [o for o in objects if o.group.split()[-1] not in hidden_layers]
    all_xyz = np.vstack([np.array(o.acoustic_geometry.vertices_attributes("xyz")) for o in objects])
    lo, hi = all_xyz.min(axis=0), all_xyz.max(axis=0)

    fig = plt.figure(figsize=(8 * len(views), 7))
    for i, (elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, len(views), i, projection="3d")
        for acoustic_object in shown:
            mesh = acoustic_object.acoustic_geometry
            polygons = [mesh.face_coordinates(face) for face in mesh.faces()]
            obstacle = acoustic_object.role == "obstacle"
            color = colors[acoustic_object.material.name]
            ax.add_collection3d(Poly3DCollection(polygons, facecolor=color, edgecolor=(0, 0, 0, 0.25 if obstacle else 0.1), linewidth=0.3, alpha=0.85 if obstacle else 0.12))
        for marker, color, points in (("*", "crimson", sources), ("o", "navy", receivers)):
            for point in points:
                ax.scatter(point.x, point.y, point.z, marker=marker, s=120, color=color, depthshade=False)
                ax.text(point.x, point.y, point.z + 0.3, point.name, color=color, fontsize=9)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    for name, color in colors.items():
        fig.axes[0].plot([], [], color=color, linewidth=6, alpha=0.6, label=name)
    fig.axes[0].legend(loc="upper left", fontsize=8, title=f"hidden: {', '.join(hidden_layers)}" if hidden_layers else None, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_plan(objects, sources, receivers, filepath):
    """A plan view of the scene's faces coloured by material, with the sources and receivers."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = _material_colors(objects)
    for acoustic_object in objects:
        color = colors[acoustic_object.material.name]
        mesh = acoustic_object.acoustic_geometry
        for face in mesh.faces():
            xyz = np.array(mesh.face_coordinates(face))
            ax.fill(xyz[:, 0], xyz[:, 1], facecolor=color, edgecolor=color, alpha=0.25 if acoustic_object.role == "obstacle" else 0.08, linewidth=0.5)
    for name, color in colors.items():
        ax.plot([], [], color=color, linewidth=6, alpha=0.5, label=name)
    for marker, color, points in (("*", "crimson", sources), ("o", "navy", receivers)):
        for point in points:
            ax.scatter(point.x, point.y, marker=marker, s=160, color=color, zorder=5)
            ax.annotate(point.name, (point.x, point.y), textcoords="offset points", xytext=(6, 6), fontsize=9, color=color)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIGURE_DPI)
    plt.close(fig)
