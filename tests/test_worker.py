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


@pytest.fixture(autouse=True)
def private_home(monkeypatch, tmp_path):
    # The records of the workers a test starts go to a folder of its own, not to the user's home
    import matplotlib

    home = tmp_path / "home"
    home.mkdir()
    # A new home would make matplotlib rebuild its font cache in every worker, which takes half a minute
    monkeypatch.setenv("MPLCONFIGDIR", matplotlib.get_configdir())
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(worker, "WORKERS", home / ".compas_acoustics" / "workers")


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


def call(service, name, token, *args):
    return json.loads(service._dispatch(name, [json.dumps({"args": [token, *args], "kwargs": {}})]))


def test_the_service_only_dispatches_the_worker_api():
    from compas_acoustics.worker.service import WorkerService

    service = WorkerService("secret")
    refused = call(service, "json.dumps", "secret", 1)
    assert refused["data"] is None and "not part of the compas_acoustics worker API" in refused["error"]
    nested = call(service, "compas_acoustics.worker.api.json.dumps", "secret", 1)
    assert nested["data"] is None and "not part of the compas_acoustics worker API" in nested["error"]
    allowed = call(service, "compas_acoustics.worker.api.info", "secret")
    assert allowed["error"] is None and allowed["data"]["compas_acoustics"] == compas_acoustics.__version__
    assert allowed["data"]["fingerprint"] == worker.fingerprint()


@pytest.mark.parametrize("token", ["wrong", "", None, 1])
def test_the_service_refuses_calls_without_the_token(token):
    from compas_acoustics.worker.service import WorkerService

    refused = call(WorkerService("secret"), "compas_acoustics.worker.api.info", token)
    assert refused["data"] is None and refused["error"] == "Unauthorised."
    bare = json.loads(WorkerService("secret")._dispatch("compas_acoustics.worker.api.info", ['{"args": [], "kwargs": {}}']))
    assert bare["data"] is None and bare["error"] == "Unauthorised."


def test_the_fingerprint_follows_the_sources(tmp_path):
    assert worker.fingerprint() == worker.fingerprint()
    assert len(worker.fingerprint()) == 64


def test_the_api_reads_and_writes_only_in_worker_folders(tmp_path):
    from compas_acoustics.worker import api

    inside = worker._folder()
    assert api._worker_folder(inside) == inside.resolve()
    assert api._worker_file(inside / "a.wav") == inside.resolve() / "a.wav"
    for outside in (tmp_path, tmp_path / "compas_acoustics_worker_x", Path.home(), inside / "nested", inside.parent):
        with pytest.raises(ValueError, match="not a compas_acoustics worker folder"):
            api._worker_folder(outside)
    with pytest.raises(ValueError, match="not a compas_acoustics worker folder"):
        api._worker_file(tmp_path / "a.wav")
    with pytest.raises(ValueError, match="not a compas_acoustics worker folder"):
        api.simulate(scene(), "pyroomacoustics", {}, None, str(tmp_path))
    with pytest.raises(ValueError, match="not a compas_acoustics worker folder"):
        api.convolve("a", {}, {}, "mono", "h.sofa", 48000, str(tmp_path / "out.wav"))


def test_a_process_is_alive_until_it_exits():
    import subprocess
    import sys

    from compas_acoustics.worker.service import process_is_alive

    assert process_is_alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not process_is_alive(child.pid)


def test_stopping_the_running_workers_shuts_them_down_and_forgets_them(tmp_path):
    import threading

    from compas.rpc import Server

    server = Server(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    (tmp_path / "1.json").write_text(json.dumps({"pid": 1, "port": server.server_address[1]}))
    (tmp_path / "2.json").write_text(json.dumps({"pid": 2, "port": 1}))
    worker.stop_running_workers(tmp_path)
    thread.join(timeout=5)
    assert not thread.is_alive()
    server.server_close()
    assert list(tmp_path.glob("*.json")) == []


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
    socket.create_connection(("127.0.0.1", running_worker.port), timeout=1).close()
    with pytest.raises(OSError):
        socket.create_connection((lan, running_worker.port), timeout=1).close()


@needs_worker
def test_the_running_worker_refuses_a_caller_without_the_token(running_worker):
    from xmlrpc.client import ServerProxy

    server = ServerProxy(f"http://127.0.0.1:{running_worker.port}")
    answer = json.loads(server.compas_acoustics.worker.api.info('{"args": [], "kwargs": {}}'))
    assert answer["data"] is None and answer["error"] == "Unauthorised."
    assert running_worker.info()["compas_acoustics"] == compas_acoustics.__version__


@needs_worker
def test_the_worker_records_itself_and_is_forgotten_when_stopped(running_worker):
    import time

    records = list(worker.WORKERS.glob("*.json"))
    mine = [json.loads(r.read_text()) for r in records]
    assert any(m["port"] == running_worker.port for m in mine)
    running_worker.stop_server()
    for _ in range(50):
        if not any(json.loads(r.read_text())["port"] == running_worker.port for r in worker.WORKERS.glob("*.json")):
            break
        time.sleep(0.1)
    else:
        pytest.fail("the worker's record is still there")


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
    data = {"ir": ir, "sampling_rate": 48000, "ambisonic_order": 1, "channel_order": "acn", "normalization": "sn3d"}
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
    assert ir.normalization is not None and ir.channel_order is not None
    assert s.impulse_responses("hybrid")


class FakeProxy:
    seen = {}
    fingerprint = None

    def __init__(self, **kwargs):
        type(self).seen = dict(kwargs, token=os.environ.get(worker.TOKEN_VARIABLE), parent=os.environ.get(worker.PARENT_VARIABLE),
                               pythonpath=os.environ.get("PYTHONPATH"))
        self.stopped = False

    def info(self, token):
        return {"compas_acoustics": compas_acoustics.__version__, "fingerprint": self.fingerprint or worker.fingerprint()}

    def stop_server(self):
        self.stopped = True


@pytest.fixture
def fake_proxy(monkeypatch):
    import compas.rpc

    monkeypatch.setattr(compas.rpc, "Proxy", FakeProxy)
    monkeypatch.setenv("COMPAS_ACOUSTICS_WORKER_PYTHON", __file__)
    monkeypatch.setattr(worker.atexit, "register", lambda *a, **k: None)
    FakeProxy.fingerprint = None
    return FakeProxy


def test_a_cold_worker_gets_time_to_start(fake_proxy):
    # The first start of a fresh environment imports the scientific stack uncached: 30 s measured, past compas.rpc's 10 s
    assert worker.proxy() is not None
    assert fake_proxy.seen["max_conn_attempts"] * 0.1 >= 60


def test_each_session_starts_its_own_worker_on_a_free_port_with_a_token(fake_proxy, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/rhino/site-packages")
    first = worker.proxy()
    seen = dict(fake_proxy.seen)
    assert seen["port"] == first.port and 0 < seen["port"] < 65536 and seen["port"] != 1753
    assert len(seen["token"]) == 32 and seen["parent"] == str(os.getpid())
    assert seen["pythonpath"] is None
    assert os.environ["PYTHONPATH"] == "/rhino/site-packages"
    assert worker.TOKEN_VARIABLE not in os.environ and worker.PARENT_VARIABLE not in os.environ
    worker._PROXY = None
    worker.proxy()
    assert fake_proxy.seen["token"] != seen["token"]


def test_a_worker_with_other_sources_is_refused_and_stopped(fake_proxy):
    fake_proxy.fingerprint = "0" * 64
    with pytest.raises(RuntimeError, match="run the installer again"):
        worker.proxy()
    assert worker._PROXY is None


def test_stop_stops_this_sessions_worker_once(fake_proxy):
    started = worker.proxy()
    worker.stop()
    assert started._proxy.stopped and worker._PROXY is None
    worker.stop()
