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
    """The broadband impulse response and Schroeder decay of every pair; ``rirs`` maps a pair label to an :class:`ImpulseResponse`."""
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


def plot_plan(objects, sources, receivers, filepath):
    """A plan view of the scene's faces coloured by material, with the sources and receivers."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {}
    cmap = plt.get_cmap("tab10")
    for acoustic_object in objects:
        name = acoustic_object.material.name
        color = colors.setdefault(name, cmap(len(colors) % 10))
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
