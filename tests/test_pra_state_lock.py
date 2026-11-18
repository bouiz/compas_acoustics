"""Thread safety of pyroomacoustics' process-wide state (PRA_STATE_LOCK)."""

import threading

import numpy as np

import compas_acoustics as ca


def _scene():
    scene = ca.Scene("lock")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    scene.add_source(ca.Source("s0", 1.5, 1.5, 1.6))
    scene.add_receiver(ca.Receiver("r", 4.0, 3.0, 1.2))
    return scene


def test_seeded_ray_tracing_in_threads_matches_the_serial_run():
    method = dict(image_source_order=1, n_rays=2000, duration=0.8, seed=7)
    serial = [next(iter(_scene().simulate(ca.Mono(**method)).values())).denormalized() for _ in range(2)]
    np.testing.assert_array_equal(serial[0], serial[1])

    results, errors = [None] * 4, []

    def run(k):
        try:
            results[k] = next(iter(_scene().simulate(ca.Mono(**method)).values())).denormalized()
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(len(results))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    for result in results:
        np.testing.assert_array_equal(result, serial[0])
