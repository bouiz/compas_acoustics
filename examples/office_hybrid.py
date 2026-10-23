"""An office, simulated two ways, compared and auralised.

Loads examples/cut_without.obj (an open-plan office, 20 x 22 x 3.9 m), simulates it with pyroomacoustics
(the mono method: image sources and ray tracing) and with the hybrid engine fitted to that result, binaural and ambisonic, plots the
reverberation times and renders speech in each format. Needs Python >= 3.11, where compas_acoustics installs the packages of
the binaural and ambisonics methods:

    pip install -e .
    python examples/office_hybrid.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt

import compas_acoustics as ca

HERE = Path(__file__).parent
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

########### materials: absorption in the octave bands 125 Hz to 8 kHz, and scattering
MATERIALS = {
    "ceiling": ca.AcousticMaterial("ceiling", [0.60, 0.62, 0.77, 0.74, 0.65, 0.81, 0.97], 0.1),
    "concrete": ca.AcousticMaterial("concrete", [0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.08], 0.05),
    "glass": ca.AcousticMaterial("glass", [0.15, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02], 0.05),
    "chairs": ca.AcousticMaterial("chairs", [0.44, 0.60, 0.77, 0.89, 0.82, 0.70, 0.58], 0.2),
    "tables": ca.AcousticMaterial("tables", [0.10, 0.10, 0.06, 0.04, 0.04, 0.03, 0.02], 0.3),
    "bookshelves": ca.AcousticMaterial("bookshelves", [0.10, 0.10, 0.06, 0.04, 0.04, 0.03, 0.02], 0.3),
}
FURNITURE = ["chairs", "tables", "bookshelves"]

########### the geometry: one acoustic object per OBJ group, all of them room boundary until told otherwise
objects = ca.AcousticObject.from_obj(HERE / "cut_without.obj")

########### the furniture stands inside the room: obstacles, reflecting from both sides
ca.AcousticObject.assign_roles_by_group(objects, {group: "obstacle" for group in FURNITURE})
ca.AcousticObject.assign_materials_by_group(objects, MATERIALS)

########### the scene: one source, one receiver facing -x
speaker = ca.Source("speaker", 12.0, 9.5, 1.5)
listener = ca.Receiver("listener", 17.0, 14.0, 1.5, direction=[-1, 0, 0])
scene = ca.Scene("office")
scene.add_acoustic_objects(objects)
scene.add_source(speaker)
scene.add_receiver(listener)

########### the reference: image sources to order 3, then ray tracing
scene.simulate(ca.Mono(image_source_order=3, ray_tracing=True, n_rays=100_000, duration=3.0, seed=42))

########### the hybrid engine: image sources early, a feedback delay network late, fitted to the reference; rendered
# binaural (stored under "binaural") and ambisonic (under "ambisonics"), the same seed giving both the same late field
scene.simulate(ca.Binaural("mono", image_source_order=2, diffraction=True, seed=42))
scene.simulate(ca.Ambisonics("mono", sh_order=1, image_source_order=2, diffraction=True, seed=42))

########### reverberation times; the analyser takes the W channel (omnidirectional) of the ambisonic response's four
warnings.filterwarnings("ignore", message="Impulse response has 4 channels")
# every method stores its responses on the scene under its own name
ir_mono = scene.get_impulse_response(speaker, listener, attribute="mono")
ir_ambisonic = scene.get_impulse_response(speaker, listener, attribute="ambisonics")
ir_binaural = scene.get_impulse_response(speaker, listener, attribute="binaural")
results = {"mono": ca.AcousticAnalyser(ir_mono).analyse(), "ambisonics": ca.AcousticAnalyser(ir_ambisonic).analyse()}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, parameter in zip(axes, ("EDT", "T30")):
    for name, marker in zip(results, "os"):
        ax.semilogx(results[name]["freqs"], results[name][parameter], marker + "-", label=name)
    ax.set(title=parameter, xlabel="octave band (Hz)")
    ax.set_xticks(results["ambisonics"]["freqs"], [f"{f:g}" for f in results["ambisonics"]["freqs"]])
    ax.minorticks_off()
axes[0].set(ylabel="time (s)", ylim=(0, None))
axes[0].legend()
plt.tight_layout()
plt.savefig(OUTPUT / "office_reverberation_times.png", dpi=150)

########### auralisation: speech through each response, in its format: mono, binaural (an HRTF per path and per delay line)
########### and ambisonic ACN/SN3D, a multichannel file for an ambisonic decoder elsewhere
speech = Path(ca.Defaults().ANECHOIC_SOUNDFILES_LIBRARY_PATH) / "female_speech.wav"
for name, ir in (("mono", ir_mono), ("binaural", ir_binaural), ("ambisonic", ir_ambisonic)):
    auraliser = ca.Auraliser("office", [ir], anechoic_sounds=[str(speech)])
    auraliser.export_convolved_sound(str(OUTPUT / f"office_speech_{name}.wav"))

print(f"written to {OUTPUT}")
plt.show()
