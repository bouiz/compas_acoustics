import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

import compas_acoustics
from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
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
        api.analyse(str(tmp_path / "ir.wav"))


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
    here = s.simulate(Pyroomacoustics(image_source_order=2, ray_tracing=False))
    there, messages = worker.simulate(s, "pyroomacoustics", {"image_source_order": 2, "ray_tracing": False})
    assert sorted(there) == sorted(here)
    for key in here:
        expected = here[key].denormalized()
        np.testing.assert_allclose(there[key].denormalized(), expected, atol=1e-6 * np.max(np.abs(expected)))
        assert there[key].sampling_rate == here[key].sampling_rate
    assert isinstance(messages, list)


@needs_worker
def test_the_hybrid_runs_in_the_worker(running_worker):
    if not running_worker.info()["hybrid"]:
        pytest.skip("the worker has no pyFDN")
    s = scene()
    rirs, _ = worker.simulate(s, "hybrid", {"tail": "eyring", "output": "ambisonic"})
    ir = next(iter(rirs.values()))
    assert ir.ambisonic_order == 1 and ir.n_channels == 4
    assert ir.normalization is not None and ir.channel_order is not None
    assert s.impulse_responses("hybrid")


@needs_worker
def test_simulate_stores_every_hybrid_output(running_worker):
    if not running_worker.info()["hybrid"]:
        pytest.skip("the worker has no hybrid extra")
    s = scene()
    rirs, _ = worker.simulate(s, "hybrid", {"tail": "eyring", "output": ["ambisonic", "binaural"]})
    assert {ir.n_channels for ir in rirs.values()} == {4}
    assert {ir.n_channels for ir in s.impulse_responses("hybrid").values()} == {4}
    binaural = s.impulse_responses("hybrid_binaural")
    assert sorted(binaural) == sorted(rirs)
    for ir in binaural.values():
        assert ir.n_channels == 2 and ir.generation["output"] == "binaural" and ir.generation["hrtf"]


def test_the_api_returns_the_responses_of_every_output_by_attribute(tmp_path, monkeypatch):
    pytest.importorskip("pyFDN")
    from compas_acoustics.worker import api

    monkeypatch.setattr(api, "_worker_folder", lambda path: Path(path))
    result = api.simulate(scene(), "hybrid", {"tail": "eyring", "output": ["binaural", "ambisonic"]}, None, str(tmp_path))
    assert list(result["responses"]) == ["hybrid_binaural", "hybrid"]
    assert all(len(paths) == 1 for paths in result["responses"].values())
    result = api.simulate(scene(), "pyroomacoustics", {"image_source_order": 1, "ray_tracing": False}, None, str(tmp_path))
    assert list(result["responses"]) == ["pyroomacoustics"]


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


def test_the_worker_install_runs_its_python_without_the_callers_pythonpath(monkeypatch, tmp_path):
    from compas_acoustics.worker import install

    calls = []
    monkeypatch.setenv("PYTHONPATH", "/rhino/site-env")
    monkeypatch.setenv("PYTHONHOME", "/rhino/home")
    monkeypatch.setattr(install, "_uv", lambda: "uv")
    monkeypatch.setattr(install, "stop_running_workers", lambda: None)
    monkeypatch.setattr(install.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs.get("env"))))
    python = install.install_worker(tmp_path / "compas_acoustics.whl", hybrid=True, home=tmp_path / "worker", config=tmp_path / "worker.json")
    assert len(calls) == 3 and all(env is not None for _, env in calls)
    assert all("PYTHONPATH" not in env and "PYTHONHOME" not in env for _, env in calls)
    assert calls[-1][0][0] == str(python)
    assert json.loads((tmp_path / "worker.json").read_text()) == {"python": str(python), "hybrid": True}


def response():
    return next(iter(scene().simulate(Pyroomacoustics(image_source_order=2, ray_tracing=False)).values()))


def test_without_a_worker_the_analysis_stays_here(monkeypatch):
    monkeypatch.setattr(worker, "proxy", lambda: pytest.fail("no worker is configured"))
    analyser, messages = worker.analyse(response())
    assert messages == [] and np.isfinite(analyser.analysis_data["STI_overall"]) and np.isfinite(analyser.analysis_data["G_overall"])


def test_sti_and_g_come_from_the_worker_where_pyrato_cannot_compute_them(monkeypatch):
    import types
    import warnings

    from compas_acoustics import AcousticAnalyser
    from compas_acoustics import analyser as analysis
    from compas_acoustics.worker import api

    ir = response()
    here = AcousticAnalyser(ir)
    here.analyse()
    # Rhino 8's Python 3.9: its pyrato < 1.0 has no STI and no G
    monkeypatch.setattr(worker, "_pyrato_has_parameters", lambda: False)
    monkeypatch.setattr(analysis, "_STI_SUPPORTED", False)
    monkeypatch.setattr(worker, "worker_python", lambda: "python3.12")
    paths = []

    def in_the_worker(path):
        # the worker's own Python 3.12, with pyrato >= 1.0
        paths.append(path)
        with monkeypatch.context() as there:
            there.setattr(analysis, "_STI_SUPPORTED", True)
            return api.analyse(path)

    monkeypatch.setattr(worker, "proxy", lambda: types.SimpleNamespace(analyse=in_the_worker))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyser, messages = worker.analyse(ir)
    assert not [w for w in caught if "STI is not available" in str(w.message)]
    assert len(paths) == 1 and api._worker_file(paths[0])
    for name in ("STI", "G"):
        assert analyser.analysis_data[name] == pytest.approx(here.analysis_data[name], abs=0.01)
        assert analyser.analysis_data[f"{name}_overall"] == pytest.approx(here.analysis_data[f"{name}_overall"], abs=0.01)
    assert analyser.analysis_data["T30"] == here.analysis_data["T30"]
    assert isinstance(messages, list)


@needs_worker
def test_the_analysis_in_the_worker_matches_the_analysis_here(running_worker, monkeypatch):
    from compas_acoustics import AcousticAnalyser

    ir = response()
    here = AcousticAnalyser(ir)
    here.analyse()
    monkeypatch.setattr(worker, "_pyrato_has_parameters", lambda: False)
    analyser, _ = worker.analyse(ir)
    for name in ("STI_overall", "G_overall"):
        assert analyser.analysis_data[name] == pytest.approx(here.analysis_data[name], abs=0.01)


@pytest.mark.parametrize("settings, normalized", [({}, True), ({"normalize": True}, True), ({"normalize": False}, False)])
def test_responses_from_the_worker_keep_the_methods_normalize(monkeypatch, settings, normalized):
    import types

    from compas_acoustics.worker import api

    monkeypatch.setattr(worker, "proxy", lambda: types.SimpleNamespace(simulate=api.simulate))
    s = scene()
    here = s.simulate(Pyroomacoustics(image_source_order=2, ray_tracing=False))
    there, _ = worker.simulate(s, "pyroomacoustics", dict(settings, image_source_order=2, ray_tracing=False))
    for key, ir in there.items():
        assert ir.normalized is normalized
        np.testing.assert_allclose(ir.denormalized(), here[key].denormalized(), atol=1e-6 * np.max(np.abs(here[key].denormalized())))


def test_the_worker_installs_the_wheel_and_nothing_else(monkeypatch, tmp_path):
    from compas_acoustics.worker import install

    calls = []
    monkeypatch.setattr(install, "_uv", lambda: "uv")
    monkeypatch.setattr(install, "stop_running_workers", lambda: None)
    monkeypatch.setattr(install.subprocess, "run", lambda command, **kwargs: calls.append(command))
    for hybrid in (False, True):
        calls.clear()
        install.install_worker(tmp_path / "compas_acoustics.whl", hybrid=hybrid, home=tmp_path / "worker", config=tmp_path / "worker.json")
        installs = [command[5:] for command in calls if command[:3] == ["uv", "pip", "install"]]
        assert installs == [[f"compas_acoustics @ {(tmp_path / 'compas_acoustics.whl').resolve().as_uri()}"]]

