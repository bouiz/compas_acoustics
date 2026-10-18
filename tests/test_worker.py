import importlib.util
import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

import compas_acoustics
from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Auraliser
from compas_acoustics import Pyroomacoustics
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics import worker

WORKER_PYTHON = os.environ.get("COMPAS_ACOUSTICS_TEST_WORKER_PYTHON")
needs_worker = pytest.mark.skipif(not WORKER_PYTHON, reason="set COMPAS_ACOUSTICS_TEST_WORKER_PYTHON to a Python >= 3.11 with compas_acoustics")
VIOLA = str(Path(compas_acoustics.__file__).parent / "anechoic_soundfiles_library" / "viola.wav")


def scene():
    s = Scene("room")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.1)))
    s.add_source(Source("src", 1.5, 1.5, 1.6))
    s.add_receiver(Receiver("rec", 5, 3, 1.5))
    return s


@pytest.fixture(autouse=True)
def no_configured_worker(monkeypatch, tmp_path):
    monkeypatch.delenv("COMPAS_ACOUSTICS_WORKER_PYTHON", raising=False)
    monkeypatch.setattr(worker, "CONFIG", tmp_path / "worker.json")
    monkeypatch.setattr(worker, "_PROXY", None)


@pytest.fixture
def running_worker(monkeypatch):
    monkeypatch.setenv("COMPAS_ACOUSTICS_WORKER_PYTHON", WORKER_PYTHON)
    proxy = worker.proxy()
    yield proxy
    proxy.stop_server()


def test_without_a_worker_everything_stays_local():
    assert worker.worker_python() is None
    assert worker.proxy() is None


def test_the_config_file_names_the_worker(tmp_path):
    worker.CONFIG.write_text(json.dumps({"python": WORKER_PYTHON or __file__}))
    assert worker.worker_python() == (WORKER_PYTHON or __file__)


def test_the_service_only_dispatches_the_worker_api():
    from compas_acoustics.worker.service import WorkerService

    refused = json.loads(WorkerService()._dispatch("json.dumps", ['{"args": [1], "kwargs": {}}']))
    assert refused["data"] is None and "not part of the compas_acoustics worker API" in refused["error"]
    nested = json.loads(WorkerService()._dispatch("compas_acoustics.worker.api.json.dumps", ['{"args": [1], "kwargs": {}}']))
    assert nested["data"] is None and "not part of the compas_acoustics worker API" in nested["error"]
    allowed = json.loads(WorkerService()._dispatch("compas_acoustics.worker.api.info", ['{"args": [], "kwargs": {}}']))
    assert allowed["error"] is None and allowed["data"]["compas_acoustics"] == compas_acoustics.__version__


def lan_address():
    # A UDP connect sends nothing; it only picks the interface a route would use
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 9))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


@needs_worker
def test_the_service_listens_on_localhost_only(running_worker):
    lan = lan_address()
    if lan is None or lan.startswith("127."):
        pytest.skip("this machine has no non-loopback address")
    socket.create_connection(("127.0.0.1", worker.PORT), timeout=1).close()
    with pytest.raises(OSError):
        socket.create_connection((lan, worker.PORT), timeout=1).close()


@needs_worker
def test_the_worker_starts_without_the_clients_python_path(tmp_path, monkeypatch):
    poisoned = tmp_path / "compas_acoustics"
    poisoned.mkdir()
    (poisoned / "__init__.py").write_text("raise ImportError('the client PYTHONPATH reached the worker')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("COMPAS_ACOUSTICS_WORKER_PYTHON", WORKER_PYTHON)
    proxy = worker.proxy()
    try:
        assert proxy.info()["compas_acoustics"] == compas_acoustics.__version__
        assert os.environ["PYTHONPATH"] == str(tmp_path)
    finally:
        proxy.stop_server()


@needs_worker
def test_simulate_in_the_worker_matches_simulating_here(running_worker):
    s = scene()
    here = s.simulate(Pyroomacoustics(image_source_order=2))
    there, messages = worker.simulate(s, "pyroomacoustics", {"image_source_order": 2})
    assert sorted(there) == sorted(here)
    for key in here:
        expected = here[key].denormalized()
        np.testing.assert_allclose(there[key].denormalized(), expected, atol=1e-6 * np.max(np.abs(expected)))
        assert there[key].sampling_rate == here[key].sampling_rate
    assert isinstance(messages, list)


def first_order_auraliser():
    ir = np.zeros((4800, 4))
    ir[100, :] = [1.0, 0.5, 0.0, 0.0]
    data = {"ir": ir, "sampling_rate": 48000, "ambisonic_order": 1, "channel_order": "acn", "normalisation": "sn3d"}
    return Auraliser("a", {"s": data}, {"s": {"filepath": VIOLA}}, output_format="binaural")


@needs_worker
def test_binaural_without_spharpy_is_decoded_in_the_worker(running_worker, monkeypatch):
    has_spharpy = importlib.util.find_spec("spharpy") is not None
    here = first_order_auraliser().convolve() if has_spharpy else None
    monkeypatch.setattr(worker, "_has_spharpy", lambda: False)
    there = worker.convolve(first_order_auraliser())
    assert there.shape[1] == 2 and np.all(np.isfinite(there))
    if here is not None:
        np.testing.assert_allclose(there, here, atol=1e-6 * np.max(np.abs(here)))


@needs_worker
def test_the_hybrid_runs_in_the_worker(running_worker):
    if not running_worker.info()["hybrid"]:
        pytest.skip("the worker has no hybrid extra")
    s = scene()
    rirs, _ = worker.simulate(s, "hybrid", {"tail": "eyring"})
    ir = next(iter(rirs.values()))
    assert ir.ambisonic_order == 1 and ir.n_channels == 4
    assert ir.normalisation is not None and ir.channel_order is not None
    assert s.impulse_responses("hybrid")


def test_a_cold_worker_gets_time_to_start(monkeypatch):
    # The first start of a fresh environment imports the scientific stack uncached: 30 s measured, past compas.rpc's 10 s
    import compas.rpc

    seen = {}

    class FakeProxy:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def info(self):
            return {"compas_acoustics": compas_acoustics.__version__}

    monkeypatch.setattr(compas.rpc, "Proxy", FakeProxy)
    monkeypatch.setenv("COMPAS_ACOUSTICS_WORKER_PYTHON", __file__)
    assert worker.proxy() is not None
    assert seen["max_conn_attempts"] * 0.1 >= 60
