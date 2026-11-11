"""The sample room of ``examples/obj.py`` read from its OBJ files, simulated and analysed end to end."""

import os

import numpy as np
import pytest

import compas_acoustics as ca

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, "examples")
SAMPLE_ROOM_GROUPS = {"floor", "plasterboard", "glass", "ceiling"}
N_RAYS = 5000


def sample_room_scene():
    concrete = ca.AcousticMaterial("concrete", [0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05], [0.01, 0.01, 0.03, 0.05, 0.1, 0.1, 0.1])
    plasterboard = ca.AcousticMaterial("plasterboard", [0.15, 0.1, 0.06, 0.04, 0.04, 0.05, 0.05], [0.01, 0.01, 0.03, 0.05, 0.2, 0.1, 0.1])
    glass = ca.AcousticMaterial("glass", [0.1, 0.07, 0.05, 0.03, 0.02, 0.02, 0.02], 0.05)
    plywood = ca.AcousticMaterial("plywood", [0.18, 0.12, 0.1, 0.09, 0.08, 0.07, 0.07], [0.05, 0.3, 0.03, 0.5, 0.6, 0.7, 0.7])

    room = ca.AcousticObject.from_obj(
        os.path.join(EXAMPLES, "sample_room.obj"),
        materials={"floor": concrete, "plasterboard": plasterboard, "glass": glass, "ceiling": concrete},
        name="sample_room",
    )
    table = ca.AcousticObject.from_obj(os.path.join(EXAMPLES, "sample_table.obj"), materials=plywood, name="sample_table")
    ca.AcousticObject.assign_roles_by_group(table, {"wood": "obstacle"})

    scene = ca.Scene(name="sample_room")
    scene.add_source(ca.Source("source_1", 1.5, 3.0, 1.5))
    scene.add_receiver(ca.Receiver("receiver_1", 2.0, 2.0, 1.0))
    scene.add_acoustic_objects(room)
    scene.add_acoustic_objects(table)
    return scene, room, table


def test_the_sample_room_obj_has_one_boundary_per_group_and_the_table_is_an_obstacle():
    _, room, table = sample_room_scene()
    assert {o.group for o in room} == SAMPLE_ROOM_GROUPS
    assert all(o.role == "boundary" and o.material is not None for o in room)
    assert [o.role for o in table] == ["obstacle"]


@pytest.mark.slow
def test_the_sample_room_simulates_and_analyses():
    scene, _, _ = sample_room_scene()
    rirs = scene.simulate(ca.Mono(ray_tracing=True, n_rays=N_RAYS, air_absorption=True))
    assert len(rirs) == 1

    ir = next(iter(rirs.values()))
    assert ir.sampling_rate == 48000
    assert np.all(np.isfinite(ir.ir)) and np.max(np.abs(ir.ir)) > 0

    results = ca.Analyser(ir).analyse()
    t30 = np.array(results["T30"], dtype=float)
    # a small, hard room: reverberant, but well under the 10 s ray-tracing cap
    assert np.isfinite(results["T30_overall"]) and 0.1 < results["T30_overall"] < 5.0
    assert np.all(t30[np.isfinite(t30)] > 0)
