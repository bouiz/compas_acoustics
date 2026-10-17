"""Regenerate every module and test ported from hybrid-rir at the pinned revision.

Run from the repository root after changing ``tests/hybrid/UPSTREAM_REVISION``:

    python tests/hybrid/regenerate.py            # every step
    python tests/hybrid/regenerate.py bands target

then review ``git diff``. Each step is one task of the integration plan
(docs/superpowers/plans/2026-10-07-hybrid-rir-integration.md). Every lookup
goes through ``_upstream`` and fails loudly when upstream changed shape.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _upstream import node  # noqa: E402
from _upstream import once  # noqa: E402
from _upstream import port  # noqa: E402
from _upstream import port_data  # noqa: E402
from _upstream import provenance  # noqa: E402
from _upstream import replace_node  # noqa: E402
from _upstream import replace_relative_import  # noqa: E402
from _upstream import span  # noqa: E402
from _upstream import upstream  # noqa: E402
from _upstream import with_provenance  # noqa: E402
from _upstream import write  # noqa: E402

HYBRID = "src/compas_acoustics/hybrid"
STEPS = {}


def step(function):
    STEPS[function.__name__] = function
    return function


# ------------------------------------------------------------- Task 11 --
@step
def bands():
    head = (
        f'"""The octave band grid of the hybrid synthesis.\n\n{provenance("src/room.py")}\n"""\n\n'
        "from __future__ import annotations\n\nimport numpy as np\nimport pyFDN\n\n"
        "from compas_acoustics.room_geometry import SYNTHESIS_BANDS\n\n"
    )
    tail = (
        "\n\n# RoomGeometry puts materials on SYNTHESIS_BANDS without importing pyFDN;\n"
        "# the two must be one grid.\n"
        "check_bands(np.asarray(SYNTHESIS_BANDS))\n"
    )
    write(f"{HYBRID}/bands.py", head + span(upstream("src/room.py"), "BAND_EDGES", "check_bands") + tail)


@step
def target():
    room = upstream("src/room.py")
    text = node(room, "RoomTarget")
    text = replace_node(text, "from_ray_tracing", "", cls="RoomTarget")
    text = once(
        text,
        '''        alpha = mean_absorption(spec.geometry, weighting=weighting)
        models = {
            "sabine": pyrato.parametric.reverberation_time_sabine,
            "eyring": pyrato.parametric.reverberation_time_eyring,
        }
        if model not in models:
            raise ValueError(f"model must be one of {sorted(models)}, got {model!r}")
        _warn_if_not_diffuse(spec.geometry, model)
        rt = np.asarray(
            models[model](
                volume=spec.geometry.volume,
                surface_area=spec.geometry.surface_area,
                mean_absorption=alpha,
                speed_of_sound=spec.speed_of_sound,
            ),
            dtype=float,
        )

        if air_absorption:
            # RT = K V / (A + 4 m V) with K = 24 ln(10) / c, so
            # 1/RT_total = 1/RT_surfaces + 4 m / K.
            k = 24.0 * np.log(10.0) / spec.speed_of_sound
            rt = 1.0 / (1.0 / rt + 4.0 * _air_absorption_coefficient() / k)
''',
        '''        alpha = mean_absorption(spec.geometry, weighting=weighting)
        # Upstream's RT computation, moved to the base package so that the
        # ray-tracing duration shares it (compas_acoustics.diffuse.predicted_rt).
        rt = predicted_rt(
            spec.geometry,
            model=model,
            weighting=weighting,
            air_absorption=air_absorption,
            speed_of_sound=spec.speed_of_sound,
        )
''',
    )
    head = (
        f'"""The decay the late field must reproduce.\n\n{provenance("src/room.py")}\n'
        "``RoomTarget`` without ``from_ray_tracing`` (the Hybrid method runs its own\n"
        "reference instead); ``from_room`` takes its reverberation time from\n"
        ":func:`compas_acoustics.diffuse.predicted_rt`.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\nfrom typing import TYPE_CHECKING\n\n"
        "import numpy as np\nimport pyFDN\nimport pyroomacoustics as pra\n\n"
        "from compas_acoustics.diffuse import _air_absorption_coefficient  # noqa: F401  (upstream's import path)\n"
        "from compas_acoustics.diffuse import mean_absorption\n"
        "from compas_acoustics.diffuse import predicted_rt\n\n"
        "from .bands import BANDS\nfrom .bands import BANDWIDTHS\nfrom .bands import N_BANDS\nfrom .bands import check_bands\n\n"
        "if TYPE_CHECKING:\n    from .pair import Pair as RoomSpec\n\n\n"
    )
    reference = '''

def reference_scale(rir: np.ndarray, n_samples: int) -> np.ndarray:
    """A pyroomacoustics response on this package's scale, ``n_samples`` long.

    pyroomacoustics spreads as ``1 / r`` and delays its response by half its
    fractional-delay kernel; this package spreads as ``1 / (4 pi r)`` with the
    direct sound on its geometric sample (upstream ``RoomSpec.ray_traced_rir``).
    """
    latency = pra.constants.get("frac_delay_length") // 2
    rir = np.asarray(rir, dtype=float).ravel()[latency:] / (4.0 * np.pi)
    return np.pad(rir, (0, max(0, n_samples - rir.size)))[:n_samples]
'''
    write(f"{HYBRID}/target.py", head + text.rstrip("\n") + "\n" + reference)


@step
def target_tests():
    port("test_bands.py")
    port(
        "test_room_target_multislope.py",
        needs_multislope=(
            "test_auto_detection_finds_one_slope_in_a_single_slope_room",
            "test_auto_detection_finds_two_slopes_in_a_coupled_room",
            "test_slopes_are_ordered_fastest_first",
            "test_inactive_slopes_are_dropped_not_carried_as_zeros",
            "test_geq_bands_selects_a_slope",
            "test_forcing_two_slopes_on_a_single_slope_room_is_documented_as_harmful",
        ),
    )
    port("test_room_target_level.py")
    # Beyond the plan's list: they pass through _compat and check the diffuse.py slice
    port("test_room_target.py")
    port("test_room_target_diffuse_field.py")


# ------------------------------------------------------------- Task 12 --
IMAGE_SOURCES = '''def image_sources(room, source_index: int, mic_index: int, spec: Pair, directivity=None) -> ImageSourceSet:
    """The image sources of one source at one microphone of a computed room.

    ``room`` has had ``image_source_model()`` run with every source and
    microphone of the scene. pyroomacoustics keeps an image when any
    microphone sees it, so the images of this pair are those visible at
    ``mic_index``. Delays, per-band gains and receiver-relative directions of
    arrival; the direct sound is order zero. The room must be built without
    air absorption, which is applied here from the path length.

    ``directivity`` is the source's pyroomacoustics directivity. It is
    evaluated along each image's emission direction from the real source,
    ``source.directions``, which is what pyroomacoustics itself uses.
    """
    source = room.sources[source_index]
    visible = np.asarray(room.visibility[source_index][mic_index], dtype=bool)
    images = np.asarray(source.images, dtype=float)[:, visible]  # (3, M)
    order = np.asarray(source.orders, dtype=int).ravel()[visible]
    wall_gain = _band_damping(source)[visible]

    offset = images - spec.receiver[:, np.newaxis]  # (3, M)
    distance = np.linalg.norm(offset, axis=0)  # (M,)
    delay = distance / spec.speed_of_sound
    spreading = 1.0 / (4.0 * np.pi * distance)

    if spec.air_absorption:
        # m is an energy attenuation coefficient, so pressure loses exp(-m d/2)
        # over the image's own path length.
        wall_gain = wall_gain * np.exp(
            -np.outer(distance, _air_absorption_coefficient()) / 2.0
        )

    gain = wall_gain * spreading[:, np.newaxis]

    if directivity is not None:
        emitted = np.asarray(source.directions[mic_index], dtype=float)[:, visible]
        azimuth = np.arctan2(emitted[1], emitted[0])
        colatitude = np.arccos(
            np.clip(emitted[2] / np.linalg.norm(emitted, axis=0), -1.0, 1.0)
        )
        response = directivity.get_response(
            azimuth=azimuth, colatitude=colatitude, degrees=False
        )
        gain = gain * np.asarray(response, dtype=float).reshape(-1, 1)

    # The rotation's columns are the receiver's axes in world coordinates, so
    # a world vector goes into the receiver frame by the transpose; the matrix
    # itself would mirror the scene left to right.
    offset = spec.receiver_rotation.T @ offset

    return ImageSourceSet(
        delay=delay,
        gain=gain,
        doa=pf.Coordinates(offset[0], offset[1], offset[2]),
        order=order,
        bands=BANDS.copy(),
        fs=spec.fs,
    )

'''


@step
def early():
    text = with_provenance(
        upstream("src/early.py"), "src/early.py",
        "The image source search moved to the caller: ``image_sources`` reads one pair\n"
        "out of a pyroomacoustics room computed for every source and receiver of a\n"
        "scene, so per-band gains always come from pyroomacoustics' damping (the\n"
        "shoebox closed form is not ported).",
    )
    text = replace_relative_import(
        text, "room",
        "from .bands import BANDS, N_BANDS, check_bands\nfrom .pair import Pair\nfrom .target import _air_absorption_coefficient\n",
    )
    for name in ("_axis_reflection_counts", "_shoebox_reflection_counts", "_shoebox_band_gains", "_damping_band_gains"):
        text = replace_node(text, name, "")
    text = replace_node(text, "image_sources", IMAGE_SOURCES)
    assert "RoomSpec" not in text and "ShoeboxGeometry" not in text
    write(f"{HYBRID}/early.py", text)


@step
def early_tests():
    port("test_early_encode.py")
    port("test_early_encode_extrapolation.py")


# ------------------------------------------------------------- Task 13 --
@step
def late():
    text = with_provenance(upstream("src/late.py"), "src/late.py", "The free paths read the faces of a RoomGeometry.")
    text = replace_relative_import(text, "room", "from .pair import Pair\nfrom .target import RoomTarget\n")
    text = replace_node(text, "_area_normals", "")
    text = once(text, "spec: RoomSpec", "spec: Pair")
    text = once(text, "_area_normals(spec.geometry)", "spec.geometry.area_normals()")
    assert "RoomSpec" not in text and "_area_normals" not in text and "ShoeboxGeometry" not in text
    write(f"{HYBRID}/late.py", text)


@step
def late_tests():
    port("test_late_design.py")
    port("test_late_hoa_defaults.py")


# ------------------------------------------------------------- Task 14 --
@step
def directional():
    scattering = upstream("src/scattering.py")
    head = (
        f'"""Ray casts against the faces of a room.\n\n{provenance("src/scattering.py")}\n'
        "Only ``_unit_normal`` and ``_first_hit``, which :mod:`directional` uses; the\n"
        "scattered image sources of that module are not ported.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "import numpy as np\n\n\n"
    )
    write(f"{HYBRID}/scattering.py", head + node(scattering, "_unit_normal") + "\n\n" + node(scattering, "_first_hit"))

    text = with_provenance(upstream("src/directional.py"), "src/directional.py")
    text = replace_relative_import(text, "room", "from .bands import N_BANDS\nfrom .pair import Pair\n")
    text = replace_relative_import(text, "scattering", "from .scattering import _first_hit\n")
    text = once(text, "_boundary_faces(spec.geometry)", "spec.geometry.faces_with_materials()")
    text = once(text, "spec: RoomSpec", "spec: Pair", count=2)
    assert "RoomSpec" not in text and "_boundary_faces" not in text
    write(f"{HYBRID}/directional.py", text)


# ------------------------------------------------------------- Task 15 --
@step
def diffraction():
    write(f"{HYBRID}/udfa.py", with_provenance(upstream("src/udfa.py"), "src/udfa.py", "No changes."))
    # btms.py: Svensson's BSD notice stays in the docstring; the provenance line is appended after it.
    write(f"{HYBRID}/btms.py", with_provenance(upstream("src/btms.py"), "src/btms.py", "No changes."))

    text = with_provenance(
        upstream("src/diffraction.py"), "src/diffraction.py",
        "Reads a RoomGeometry, which carries upstream MeshGeometry's attribute names.",
    )
    text = replace_relative_import(
        text, "room",
        "from compas_acoustics.room_geometry import OBSTACLE_OFFSET, RoomGeometry, _polygon_area_and_centroid\n\n"
        "from .bands import BANDS\nfrom .pair import Pair\nfrom .target import _air_absorption_coefficient\n",
    )
    # upstream checks for obstacles once per model (diffraction_paths, btms_paths)
    text = once(
        text,
        'if not isinstance(geometry, MeshGeometry) or not geometry.obstacle_faces:\n'
        '        raise ValueError("diffraction needs a MeshGeometry with obstacles")',
        'if not geometry.obstacle_faces:\n'
        '        raise ValueError("diffraction needs surfaces with role=\'obstacle\'")',
        count=2,
    )
    text = text.replace("MeshGeometry", "RoomGeometry").replace("RoomSpec", "Pair")
    write(f"{HYBRID}/diffraction.py", text)


@step
def diffraction_tests():
    for name in ("udfa_single_oracle.mat", "double_edge_oracle.mat", "btms_first.mat", "btms_second.mat", "btms_plate.mat", "btms_fd.mat"):
        port_data(name)
    port("test_udfa.py")
    port("test_udfa_oracle.py")
    port("test_btms.py")
    port("test_btms_oracle.py")
    port("test_diffraction.py")


# ------------------------------------------------------------- Task 16 --
@step
def envelope():
    write(f"{HYBRID}/envelope.py", with_provenance(upstream("src/envelope.py"), "src/envelope.py", "No changes."))


@step
def envelope_tests():
    port("test_envelope.py")


if __name__ == "__main__":
    for name in sys.argv[1:] or STEPS:
        print(f"regenerate: {name}")
        STEPS[name]()
