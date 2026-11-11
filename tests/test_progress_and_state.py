"""Progress reports, cancellation, thread safety of pyroomacoustics' global state, and the lazy package import."""

import subprocess
import sys
import threading

import numpy as np
import pytest

import compas_acoustics as ca
from compas_acoustics.progress import Cancelled
from compas_acoustics.progress import reporting


def _scene(n_sources=1):
    scene = ca.Scene("progress")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    for k in range(n_sources):
        scene.add_source(ca.Source(f"s{k}", 1.5 + k, 1.5, 1.6))
    scene.add_receiver(ca.Receiver("r", 4.0, 3.0, 1.2))
    return scene


def test_reporting_sees_every_stage_and_restores_the_outer_callback():
    outer, inner = [], []
    with reporting(lambda *report: outer.append(report)):
        with reporting(lambda *report: inner.append(report)):
            _scene(2).simulate(ca.Mono(image_source_order=1, ray_tracing=False))
        _scene().simulate(ca.Mono(image_source_order=1, ray_tracing=False))
    assert inner[0] == ("image sources", None, None)
    assert [r for r in inner if r[0] == "responses"] == [("responses", 0, 2), ("responses", 1, 2)]
    # the inner block's reports did not reach the outer callback
    assert outer == [("image sources", None, None), ("responses", 0, 1)]


def test_raising_from_the_callback_stops_the_simulation():
    def stop(stage, done, of):
        if stage == "responses":
            raise Cancelled()

    with pytest.raises(Cancelled):
        with reporting(stop):
            _scene().simulate(ca.Mono(image_source_order=1, ray_tracing=False))
    # nothing is left behind: the next run reports and finishes
    _scene().simulate(ca.Mono(image_source_order=1, ray_tracing=False))


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


def test_the_package_import_is_lazy():
    code = (
        "import sys, compas_acoustics as ca;"
        "heavy = [m for m in ('pyroomacoustics', 'pyfar', 'pyrato', 'compas', 'pyFDN', 'torch') if m in sys.modules];"
        "print(','.join(heavy));"
        "ca.Mono; print('pyroomacoustics' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[0] == ""
    assert out[1] == "True"


def test_public_names_resolve():
    for name in ca.__all__:
        assert getattr(ca, name) is not None
    from compas_acoustics import Mono  # noqa: F401
    from compas_acoustics import Source  # noqa: F401

    assert "Mono" in dir(ca)
    with pytest.raises(AttributeError):
        ca.NotAName  # noqa: B018
