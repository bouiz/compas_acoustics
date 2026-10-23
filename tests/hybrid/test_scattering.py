"""Surface scattering in the hybrid (Ewert et al. 2025) against RAZR 0.96.3's method (data/razr_scattering)."""

import json
import sys
from pathlib import Path

import numpy as np
import pyfar as pf
import pytest
from scipy.spatial import cKDTree

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.hybrid.bands import BANDS
from compas_acoustics.hybrid.early import ImageSourceSet
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.early import route_to_directions
from compas_acoustics.hybrid.late import design_fdn
from compas_acoustics.hybrid.late import directional_delays
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.hybrid.late import render
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.hybrid.scattering import allpass_delays
from compas_acoustics.hybrid.scattering import complementary_scattering
from compas_acoustics.hybrid.scattering import scattering_decay_time
from compas_acoustics.hybrid.scattering import scattering_weights
from compas_acoustics.hybrid.scattering import smear
from compas_acoustics.hybrid.synthesis import prepare
from compas_acoustics.hybrid.ambisonics import render_ambisonics
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.room.pra import receivers_to_pra_mic_array
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.geometry import pra_octave_grid
from compas_acoustics.simulation_methods import Ambisonics

sys.path.insert(0, str(Path(__file__).parent / "data"))
from hybrid_reference import reference_rir  # noqa: E402
from hybrid_reference import reference_scene  # noqa: E402

FS = 48000
DATA = Path(__file__).parent / "data"
ORACLE = json.loads((DATA / "razr_scattering" / "razr_scattering_oracle.json").read_text())
CASES = ORACLE["cases"]
ORDER = ORACLE["options"]["ism_order"]
HIGH = 7  # the 8 kHz octave; the oracle's ratios are above 5 kHz
WALLS = ("-z", "-y", "-x", "+x", "+y", "+z")  # RAZR's wall order
(SMART,) = [c for c in ORACLE["coupling"] if c["fdn_smart_ch_mapping"] == 1]


def case_id(case):
    return f"room{case['room']}-d{case['d']}"


def scene(case, scattering):
    """An oracle room; ``scattering`` is one value, or one per wall in RAZR's order."""
    s = Scene("oracle")
    walls = AcousticObjectsLibrary.shoebox_room(*case["boxsize"])
    per_wall = np.broadcast_to(scattering, 6)
    for wall in walls:
        xyz = np.array(wall.acoustic_geometry.vertices_attributes("xyz"))
        axis = int(np.argmin(np.ptp(xyz, axis=0)))
        name = ("+" if xyz[0, axis] > 0.0 else "-") + "xyz"[axis]
        wall.material = AcousticMaterial(f"wall{name}", case["absorption"], float(per_wall[WALLS.index(name)]))
    s.add_acoustic_objects(walls)
    s.add_source(Source("src", *case["source"]))
    s.add_receiver(Receiver("rec", *case["receiver"]))
    return s


def pair(s):
    geometry = RoomGeometry.from_scene(s)
    receiver = s.receivers[0]
    return Pair(geometry, s.sources[0].position, receiver.position, receiver.rotation, float(FS), 343.0, False)


def computed(geometry, s, ray_tracing):
    with pra_octave_grid():
        room = geometry.to_pra(FS, ORDER, ray_tracing=ray_tracing)
        room.add_soundsource(source_to_pra(s.sources[0]))
        room.add_microphone_array(receivers_to_pra_mic_array(s.receivers, sampling_rate=FS))
        room.image_source_model()
    return room


def split(case, scattering):
    """``(spec, full, parts)``: the images without scattering, and split into specular and scattered parts."""
    s = scene(case, scattering)
    spec = pair(s)
    full = image_sources(computed(spec.geometry, s, False), 0, 0, spec)
    parts = image_sources(
        computed(spec.geometry, s, True), 0, 0, spec,
        scattered_room=computed(complementary_scattering(spec.geometry), s, True),
    )
    assert np.array_equal(full.order, parts.order)
    assert np.allclose(full.delay, parts.delay)
    return spec, full, parts


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_specular_and_scattered_shares_match_razr_per_order(case):
    _, full, parts = split(case, case["d"])
    for k in range(1, ORDER + 1):
        reference = np.sum(full.gain[full.order == k, HIGH] ** 2)
        specular = np.sum(parts.gain[parts.order == k, HIGH] ** 2) / reference
        scattered = np.sum(parts.scattered[parts.order == k, HIGH] ** 2) / reference
        assert specular == pytest.approx(case["specular_ratio"][k - 1], rel=0.02)
        assert scattered == pytest.approx(case["scattered_ratio"][k - 1], rel=0.02)


def test_specular_and_scattered_shares_match_razr_per_wall():
    probe = ORACLE["per_wall"]
    spec, full, parts = split(CASES[0], probe["d"])
    # Image positions in room coordinates: the receiver-frame offsets rotated back
    ours = spec.receiver + full.doa.cartesian @ spec.receiver_rotation.T
    distance, match = cKDTree(ours).query(np.asarray(probe["positions"], dtype=float))
    assert np.all(distance < 1e-4)  # pyroomacoustics computes images in float32
    assert np.array_equal(full.order[match], probe["order"])
    specular = (parts.gain[match, HIGH] / full.gain[match, HIGH]) ** 2
    scattered = (parts.scattered[match, HIGH] / full.gain[match, HIGH]) ** 2
    assert np.allclose(specular, probe["specular_ratio"], rtol=0.02)
    assert np.allclose(scattered, probe["scattered_ratio"], rtol=0.02, atol=1e-12)


@pytest.mark.parametrize("s", [0.0, 1.0])
def test_full_and_no_scattering_are_exact(s):
    _, full, parts = split(CASES[0], s)
    reflected = parts.order >= 1
    if s == 1.0:
        assert np.all(parts.gain[reflected] == 0.0)
        assert np.allclose(parts.scattered[reflected], full.gain[reflected], rtol=1e-12)
    else:
        assert np.allclose(parts.gain, full.gain, rtol=1e-12)
        assert np.all(parts.scattered == 0.0)
    assert np.all(parts.scattered[~reflected] == 0.0)
    assert np.allclose(parts.gain[~reflected], full.gain[~reflected], rtol=1e-12)


def test_allpass_delays_match_razr():
    for case in CASES:
        delays = allpass_delays(scattering_decay_time(case["mean_distance"], 343.0), FS)
        assert np.array_equal(delays, case["allpass_delays"]), case_id(case)


def test_smear_is_an_allpass_with_the_papers_delays():
    expected = np.round(0.2 * np.log10(np.sqrt(2)) / 3 * 48000 / np.pi ** np.arange(4))
    assert np.allclose(allpass_delays(0.2, 48000), expected)
    impulse = np.zeros((48000, 2))
    impulse[0] = 1.0
    smeared = smear(impulse, 0.2, 48000)
    assert smeared.shape == impulse.shape
    assert np.sum(smeared[:, 0] ** 2) == pytest.approx(1.0, rel=1e-3)
    assert np.array_equal(smeared[:, 0], smeared[:, 1])


def unit(xyz):
    xyz = np.asarray(xyz, dtype=float)
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def rule(arrival, lines):
    """What RAZR's G shows with fdn_smart_ch_mapping: clipped dot products, unit l2 norm per row."""
    w = np.clip(arrival @ lines.T, 0.0, None)
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def test_razrs_coupling_is_the_rule():
    assert SMART["residual"] < 1e-6
    azim, elev = np.radians(SMART["azim"]), np.radians(SMART["elev"])
    lines = np.c_[np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    arrival = unit(SMART["relpos"])
    order = np.asarray(SMART["order"])
    driving = np.asarray(SMART["is_fdn_input"], dtype=bool)
    scattered = np.asarray(SMART["scattered_gains"])
    specular = np.asarray(SMART["specular_gains"])
    assert np.allclose(scattered[order >= 1], rule(arrival[order >= 1], lines), atol=1e-6)
    assert np.all(scattered[order == 0] == 0.0)
    assert np.allclose(specular, rule(-arrival[driving], lines), atol=1e-6)


def test_coupling_follows_razr():
    """RAZR's rule on compas' lines, relative to compas' drive, for the oracle's images: in RAZR a scattered
    part's weights and a last-order specular image's both have unit l2 norm."""
    relpos = np.asarray(SMART["relpos"], dtype=float)[np.asarray(SMART["order"]) >= 1]
    razr_ratio = (np.linalg.norm(np.asarray(SMART["scattered_gains"]), axis=1).max()
                  / np.linalg.norm(np.asarray(SMART["specular_gains"]), axis=1).max())
    directions = fdn_directions()
    iss = ImageSourceSet(
        delay=np.linalg.norm(relpos, axis=1) / 343.0,
        gain=np.ones((relpos.shape[0], BANDS.size)),
        doa=pf.Coordinates(relpos[:, 0], relpos[:, 1], relpos[:, 2]),
        order=np.ones(relpos.shape[0], dtype=int),
        bands=BANDS.copy(),
        fs=float(FS),
    )
    weights = scattering_weights(iss.doa, directions)
    drive = route_to_directions(iss, directions, opposite=True)
    assert np.allclose(weights, rule(unit(relpos), unit(directions.cartesian)), rtol=1e-6, atol=1e-12)
    ratio = np.linalg.norm(weights, axis=1) / np.linalg.norm(drive, axis=1)
    assert np.allclose(ratio, razr_ratio, rtol=1e-6)


def test_tap_is_heard_through_the_lines_attenuation_filters():
    """RAZR places a scattered part in a line ahead by the line's delay, so it passes the line's absorption
    filter before it is heard (oracle, line_filter): the tap equals driving that line one delay earlier."""
    spec = pair(scene(CASES[1], 0.1))
    directions = fdn_directions()
    builds = design_fdn(RoomTarget.from_room(spec, model="eyring"), [directional_delays(spec, directions)])
    line = 3
    delay = int(np.asarray(builds[0].delays)[line])
    tap = np.zeros((6000, directions.csize))
    tap[delay + 50, line] = 1.0
    drive = np.zeros_like(tap)
    drive[50, line] = 1.0
    assert np.allclose(render(builds, np.zeros_like(tap), tap=tap), render(builds, drive), atol=1e-12)


def test_zero_scattering_changes_nothing():
    kwargs = dict(tail="eyring", image_source_order=2, duration=0.3, air_absorption=False)
    off = next(iter(scene(CASES[0], 0.0).simulate(Ambisonics(**kwargs)).values()))
    on = next(iter(scene(CASES[0], 0.0).simulate(Ambisonics(scattering=True, **kwargs)).values()))
    assert np.allclose(np.asarray(on.ir), np.asarray(off.ir), rtol=0.0, atol=1e-12 * np.abs(off.ir).max())
    assert on.generation["scattering"] is True


def test_scattering_moves_energy_from_the_early_to_the_late_field():
    kwargs = dict(tail="eyring", image_source_order=2, duration=0.5, air_absorption=False)
    off, on = Ambisonics(**kwargs), Ambisonics(scattering=True, **kwargs)
    scene(CASES[0], CASES[0]["d"]).simulate(off)
    scene(CASES[0], CASES[0]["d"]).simulate(on)
    (a,), (b,) = off.results.values(), on.results.values()
    assert np.all(np.isfinite(b.rir))
    assert np.sum(b.early[:, 0] ** 2) < np.sum(a.early[:, 0] ** 2)
    assert b.core.images.scattered is not None and b.core.images.wall is not None


def test_without_scattering_nothing_changes():
    """``scattering=False`` gives the response stored at 227725f (data/hybrid_reference.py), as the default does."""
    before = np.load(DATA / "hybrid_reference_227725f.npz")["rir"]
    after = reference_rir(scattering=False)
    assert np.any(after[-200:] != 0.0)  # the late field is inside the window
    np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-12 * np.abs(before).max())
    assert np.array_equal(reference_rir(), after)


def test_scattering_is_off_by_default():
    hybrid = Ambisonics(tail="eyring", image_source_order=1, duration=0.05)
    (response,) = reference_scene().simulate(hybrid).values()
    assert response.generation["scattering"] is False
    (result,) = hybrid.results.values()
    assert result.core.images.scattered is None


def test_scattering_needs_the_scattered_parts():
    s = scene(CASES[1], 0.1)
    spec = pair(s)
    iss = image_sources(computed(spec.geometry, s, False), 0, 0, spec)
    with pytest.raises(ValueError, match="scattered"):
        render_ambisonics(prepare(spec, iss, RoomTarget.from_room(spec, model="eyring"), max_order=1, duration_s=0.2, scattering=True))


def test_prepare_assembles_the_tap_from_the_scattered_parts_of_orders_one_to_the_last():
    """The late lines equal the network driven by the specular last order plus the tap built here from the public pieces."""
    from dataclasses import replace as replace_fields

    from compas_acoustics.hybrid.directional import directional_absorption_filters
    from compas_acoustics.hybrid.early import render_images

    case, last, duration, speed, drive_gain = CASES[1], 2, 0.3, 331.0, 1.7
    spec, _, parts = split(case, 0.3)
    spec = replace_fields(spec, speed_of_sound=speed)
    target = RoomTarget.from_room(spec, model="eyring")
    result = render_ambisonics(prepare(spec, parts, target, max_order=last, duration_s=duration, scattering=True, drive_gain=drive_gain))

    def subset(mask):
        xyz = parts.doa.cartesian[mask]
        return replace_fields(
            parts, delay=parts.delay[mask], gain=parts.gain[mask], order=parts.order[mask],
            doa=pf.Coordinates(xyz[:, 0], xyz[:, 1], xyz[:, 2]), scattered=parts.scattered[mask], wall=parts.wall[mask])

    def lines(tap_orders):
        n_samples = int(np.ceil(duration * FS))
        directions = fdn_directions()
        delays = [directional_delays(spec, directions, rng=slope) for slope in range(target.n_slopes)]
        builds = design_fdn(target, delays, rng=0)
        driving = subset(parts.order == last)
        drive = render_images(driving, scattering_weights(driving.doa, directions, opposite=True), n_samples)
        reflected = subset(np.isin(parts.order, tap_orders))
        x, y, z = case["boxsize"]
        free_path = 4.0 * (x * y * z) / (2.0 * (x * y + y * z + x * z))
        tap = smear(
            render_images(replace_fields(reflected, gain=reflected.scattered), scattering_weights(reflected.doa, directions), n_samples),
            scattering_decay_time(free_path, speed), FS,
        ) * drive_gain
        return render(builds, drive * drive_gain, tap=tap, line_eq=directional_absorption_filters(spec, directions))

    expected = lines(range(1, last + 1))
    assert np.max(np.abs(expected)) > 0.0
    np.testing.assert_allclose(result.late_lines, expected, rtol=0.0, atol=1e-12 * np.max(np.abs(expected)))
    without_last = lines(range(1, last))
    assert np.max(np.abs(without_last - expected)) > 1e-3 * np.max(np.abs(expected))
