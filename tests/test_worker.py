import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

import compas_acoustics
from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Mono
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics import analyser
from compas_acoustics import worker

WORKER_PYTHON = os.environ.get("COMPAS_ACOUSTICS_TEST_WORKER_PYTHON")
needs_worker = pytest.mark.skipif(not WORKER_PYTHON, reason="set COMPAS_ACOUSTICS_TEST_WORKER_PYTHON to a Python >= 3.11 with compas_acoustics")
# the local reference needs STI and G, which pyrato < 1.0 (Python < 3.11, Rhino 8) cannot compute
needs_sti = pytest.mark.skipif(not analyser._STI_SUPPORTED, reason="STI and G need pyrato >= 1.0 (Python >= 3.11)")
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
    monkeypatch.setattr(worker, "_CLIENT", None)


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
    client = worker.client()
    yield client
    client.stop()


def test_without_a_worker_everything_stays_local():
    assert worker.worker_python() is None
    assert worker.client() is None


def test_the_config_file_names_the_worker(tmp_path):
    worker.CONFIG.write_text(json.dumps({"python": WORKER_PYTHON or __file__}))
    assert worker.worker_python() == (WORKER_PYTHON or __file__)


def call(function, *args, token="secret", authorization=None, **kwargs):
    """The service's answer to a POST of ``function``, decoded: (status, payload)."""
    from compas_pb import pb_dump_bts
    from compas_pb import pb_load_bts

    from compas_acoustics.worker.service import dispatch

    header = authorization if authorization is not None else f"Bearer {token}"
    status, body = dispatch("secret", header, function, pb_dump_bts({"args": list(args), "kwargs": kwargs}))
    answer = pb_load_bts(body)
    return status, answer["result"] if status == 200 else answer


def test_the_service_only_dispatches_the_worker_api():
    for function in ("dumps", "json", "_identity", "warnings", "importlib"):
        status, answer = call(function)
        assert status == 404 and "not part of the compas_acoustics worker API" in answer["error"]
    status, answer = call("info")
    assert status == 200 and answer["compas_acoustics"] == compas_acoustics.__version__
    assert answer["fingerprint"] == worker.fingerprint()


def test_info_names_the_schema_compas_pb_and_the_speed_of_sound():
    import compas_pb

    import compas_acoustics.pb
    from compas_acoustics import Defaults

    _, answer = call("info")
    assert answer["schema"] == compas_acoustics.pb.SCHEMA
    assert answer["compas_pb"] == compas_pb.__version__
    assert answer["speed_of_sound"] == Defaults().DEFAULT_SPEED_OF_SOUND


@pytest.mark.parametrize("authorization", ["Bearer wrong", "Bearer ", "", "secret", "Basic secret"])
def test_the_service_refuses_calls_without_the_token(authorization):
    status, answer = call("info", authorization=authorization)
    assert status == 401 and answer == {"error": "Unauthorised."}


def test_a_body_that_is_not_a_compas_pb_payload_is_a_400():
    from compas_pb import pb_load_bts

    from compas_acoustics.worker.service import dispatch

    status, body = dispatch("secret", "Bearer secret", "info", b"not protobuf")
    assert status == 400 and "compas_pb" in pb_load_bts(body)["error"]


def test_an_unknown_method_names_the_three_methods():
    status, answer = call("simulate", scene(), "hybrid", {})
    assert status == 400 and answer["error"] == "Unknown method 'hybrid': 'binaural', 'ambisonics' or 'mono'."


def test_an_unexpected_error_is_a_500_with_a_traceback(monkeypatch):
    from compas_acoustics.worker import api

    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "info", broken)
    status, answer = call("info")
    assert status == 500 and "Traceback" in answer["error"] and answer["error"].rstrip().endswith("RuntimeError: boom")


def test_results_travel_as_compas_pb_with_numpy_made_plain():
    status, answer = call("simulate", scene(), "mono", {"image_source_order": 1, "ray_tracing": False})
    assert status == 200
    (attribute, responses), = answer["responses"].items()
    ir = next(iter(responses.values()))
    assert attribute == "mono" and ir.denormalized().ndim == 1
    assert isinstance(answer["warnings"], list)


def test_the_fingerprint_follows_the_sources(tmp_path):
    assert worker.fingerprint() == worker.fingerprint()
    assert len(worker.fingerprint()) == 64


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

    from compas_acoustics.worker.service import WorkerService

    server = WorkerService(("127.0.0.1", 0), "secret")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    (tmp_path / "1.json").write_text(json.dumps({"pid": 1, "port": server.server_address[1], "token": "secret"}))
    (tmp_path / "2.json").write_text(json.dumps({"pid": 2, "port": 1, "token": "x"}))
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
    import urllib.error
    import urllib.request

    from compas_pb import pb_dump_bts

    request = urllib.request.Request(f"http://127.0.0.1:{running_worker.port}/api/info", data=pb_dump_bts({"args": [], "kwargs": {}}),
                                     method="POST", headers={"Content-Type": "application/x-protobuf"})
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=5)
    assert refused.value.code == 401
    with urllib.request.urlopen(f"http://127.0.0.1:{running_worker.port}/ping", timeout=5) as pong:
        assert pong.status == 200
    assert running_worker.info()["compas_acoustics"] == compas_acoustics.__version__


@needs_worker
def test_the_worker_records_itself_and_is_forgotten_when_stopped(running_worker):
    import time

    records = list(worker.WORKERS.glob("*.json"))
    mine = [json.loads(r.read_text()) for r in records]
    assert any(m["port"] == running_worker.port for m in mine)
    running_worker.stop()
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
    client = worker.client()
    try:
        assert client.info()["compas_acoustics"] == compas_acoustics.__version__
        assert os.environ["PYTHONPATH"] == str(tmp_path)
    finally:
        client.stop()


@needs_worker
def test_simulate_in_the_worker_matches_simulating_here(running_worker):
    s = scene()
    here = s.simulate(Mono(image_source_order=2, ray_tracing=False))
    there, messages = worker.simulate(s, "mono", {"image_source_order": 2, "ray_tracing": False})
    assert sorted(there) == sorted(here)
    for key in here:
        expected = here[key].denormalized()
        np.testing.assert_allclose(there[key].denormalized(), expected, atol=1e-6 * np.max(np.abs(expected)))
        assert there[key].sampling_rate == here[key].sampling_rate
    assert isinstance(messages, list)


@needs_worker
def test_ambisonics_runs_in_the_worker(running_worker):
    if not running_worker.info()["hybrid"]:
        pytest.skip("the worker has no pyFDN")
    s = scene()
    rirs, _ = worker.simulate(s, "ambisonics", {"tail": "eyring"})
    ir = next(iter(rirs.values()))
    assert ir.ambisonic_order == 1 and ir.n_channels == 4
    assert ir.normalization is not None and ir.channel_order is not None
    assert s.impulse_responses("ambisonics") == rirs


@needs_worker
def test_binaural_runs_in_the_worker(running_worker):
    if not running_worker.info()["hybrid"]:
        pytest.skip("the worker has no pyFDN")
    s = scene()
    rirs, _ = worker.simulate(s, "binaural", {"tail": "eyring"})
    assert s.impulse_responses("binaural") == rirs
    for ir in rirs.values():
        assert ir.n_channels == 2 and ir.format == "binaural" and ir.generation["hrtf"]


def test_the_api_returns_the_responses_under_the_methods_name():
    pytest.importorskip("pyFDN")
    from compas_acoustics import ImpulseResponse
    from compas_acoustics.worker import api

    for method in ("binaural", "ambisonics"):
        result = api.simulate(scene(), method, {"tail": "eyring"})
        assert list(result["responses"]) == [method] and "formats" not in result
        assert all(isinstance(ir, ImpulseResponse) for ir in result["responses"][method].values())
    result = api.simulate(scene(), "mono", {"image_source_order": 1, "ray_tracing": False})
    assert list(result["responses"]) == ["mono"]


def test_references_reach_the_method_as_responses():
    from compas_acoustics.simulation_methods import SimulationContext
    from compas_acoustics.worker import api

    s = scene()
    reference = s.simulate(Mono(image_source_order=1, ray_tracing=False))
    key = next(iter(reference))
    assert key == SimulationContext.key(s.sources[0], s.receivers[0])
    seen = {}

    class Recording(Mono):
        def __init__(self, reference=None, **kwargs):
            seen["reference"] = reference
            super().__init__(**kwargs)

    api.METHODS["recording"] = Recording
    try:
        api.simulate(s, "recording", {"image_source_order": 1, "ray_tracing": False}, reference)
    finally:
        del api.METHODS["recording"]
    assert list(seen["reference"]) == [key]
    np.testing.assert_array_equal(seen["reference"][key].denormalized(), reference[key].denormalized())


class FakeProcess:
    def __init__(self):
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


class FakeWorker:
    """Stands in for the worker process and its HTTP service."""

    def __init__(self):
        self.spawned = []
        self.version = compas_acoustics.__version__
        self.fingerprint = None
        self.pings_before_up = 0
        self.shutdowns = 0

    def spawn(self, python, port, env, log):
        self.spawned.append({"python": python, "port": port, "env": dict(env)})
        self.process = FakeProcess()
        return self.process

    def send(self, client, path, body=b"", authorize=True, headers=None):
        from compas_pb import pb_dump_bts

        if path == "/ping":
            if self.pings_before_up:
                self.pings_before_up -= 1
                raise ConnectionRefusedError()
            return 200, b"pong"
        if path == "/shutdown":
            self.shutdowns += 1
            return 200, b""
        assert path == "/api/info"
        return 200, pb_dump_bts({"result": {"compas_acoustics": self.version, "fingerprint": self.fingerprint or worker.fingerprint()}})


@pytest.fixture
def fake_worker(monkeypatch):
    fake = FakeWorker()
    monkeypatch.setattr(worker, "_spawn", fake.spawn)
    monkeypatch.setattr(worker.Client, "_send", lambda self, path, body=b"", authorize=True, headers=None: fake.send(self, path, body, authorize))
    monkeypatch.setattr(worker, "PING_SECONDS", 0)
    monkeypatch.setenv("COMPAS_ACOUSTICS_WORKER_PYTHON", __file__)
    monkeypatch.setattr(worker.atexit, "register", lambda *a, **k: None)
    return fake


def test_a_cold_worker_gets_time_to_start(fake_worker):
    # The first start of a fresh environment imports the scientific stack uncached: 30 s measured
    assert worker.CONNECTION_ATTEMPTS * 0.1 >= 60
    fake_worker.pings_before_up = 5
    assert worker.client() is not None


def test_each_session_starts_its_own_worker_on_a_free_port_with_a_token(fake_worker, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/rhino/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/rhino/home")
    first = worker.client()
    seen = fake_worker.spawned[-1]
    assert seen["port"] == first.port and 0 < seen["port"] < 65536
    env = seen["env"]
    assert len(env[worker.TOKEN_VARIABLE]) == 32 and env[worker.PARENT_VARIABLE] == str(os.getpid())
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert os.environ["PYTHONPATH"] == "/rhino/site-packages"
    assert worker.TOKEN_VARIABLE not in os.environ and worker.PARENT_VARIABLE not in os.environ
    worker._CLIENT = None
    worker.client()
    assert fake_worker.spawned[-1]["env"][worker.TOKEN_VARIABLE] != env[worker.TOKEN_VARIABLE]


def test_a_worker_with_other_sources_is_refused_and_stopped(fake_worker):
    fake_worker.fingerprint = "0" * 64
    with pytest.raises(RuntimeError, match="run the installer again"):
        worker.client()
    assert worker._CLIENT is None
    assert fake_worker.shutdowns == 1


def test_a_worker_that_dies_while_starting_says_so(fake_worker, monkeypatch):
    fake_worker.pings_before_up = 10**6
    monkeypatch.setattr(FakeProcess, "poll", lambda self: 1)
    with pytest.raises(RuntimeError, match="exited"):
        worker.client()


def test_stop_stops_this_sessions_worker_once(fake_worker):
    worker.client()
    worker.stop()
    assert fake_worker.shutdowns == 1 and worker._CLIENT is None
    worker.stop()
    assert fake_worker.shutdowns == 1


def test_an_error_answer_raises_a_worker_error_with_its_text(fake_worker, monkeypatch):
    from compas_pb import pb_dump_bts

    started = worker.client()
    monkeypatch.setattr(worker.Client, "_send", lambda self, path, body=b"", authorize=True, headers=None: (400, pb_dump_bts({"error": "bad input"})))
    with pytest.raises(worker.WorkerError, match="bad input") as error:
        started.simulate(scene(), "x", {})
    assert error.value.status == 400


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
    return next(iter(scene().simulate(Mono(image_source_order=2, ray_tracing=False)).values()))


@needs_sti
def test_without_a_worker_the_analysis_stays_here(monkeypatch):
    monkeypatch.setattr(worker, "client", lambda: pytest.fail("no worker is configured"))
    analyser, messages = worker.analyse(response())
    assert messages == [] and np.isfinite(analyser.analysis_data["STI_overall"]) and np.isfinite(analyser.analysis_data["G_overall"])


@needs_sti
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
    sent = []

    def in_the_worker(response):
        # the worker's own Python 3.12, with pyrato >= 1.0
        sent.append(response)
        with monkeypatch.context() as there:
            there.setattr(analysis, "_STI_SUPPORTED", True)
            return api.analyse(response)

    monkeypatch.setattr(worker, "client", lambda: types.SimpleNamespace(analyse=in_the_worker))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyser, messages = worker.analyse(ir)
    assert not [w for w in caught if "STI is not available" in str(w.message)]
    assert sent == [ir]
    for name in ("STI", "G"):
        assert analyser.analysis_data[name] == pytest.approx(here.analysis_data[name], abs=0.01)
        assert analyser.analysis_data[f"{name}_overall"] == pytest.approx(here.analysis_data[f"{name}_overall"], abs=0.01)
    assert analyser.analysis_data["T30"] == here.analysis_data["T30"]
    assert isinstance(messages, list)


@needs_sti
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


@needs_sti
def test_the_worker_computes_only_sti_and_g(monkeypatch):
    from compas_acoustics import AcousticAnalyser
    from compas_acoustics.worker import api

    ir = response()
    for name in ("reverberation_times", "clarity_definition", "direct_to_reverberant_ratio", "center_time"):
        monkeypatch.setattr(AcousticAnalyser, name, lambda *args, **kwargs: pytest.fail("the client computes these itself"))
    assert set(api.analyse(ir)["analysis"]) == {"STI", "STI_overall", "G", "G_overall"}


@pytest.mark.parametrize("settings, normalized", [({}, True), ({"normalize": True}, True), ({"normalize": False}, False)])
def test_responses_from_the_worker_keep_the_methods_normalize(monkeypatch, settings, normalized):
    import types

    from compas_acoustics.worker import api

    monkeypatch.setattr(worker, "client", lambda: types.SimpleNamespace(simulate=api.simulate))
    s = scene()
    here = s.simulate(Mono(image_source_order=2, ray_tracing=False))
    there, _ = worker.simulate(s, "mono", dict(settings, image_source_order=2, ray_tracing=False))
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



def test_an_empty_or_zero_result_still_travels(monkeypatch):
    from compas_acoustics.worker import api

    for value in ([], {}, 0, 0.0, None, ""):
        monkeypatch.setattr(api, "info", lambda value=value: value)
        status, answer = call("info")
        assert status == 200 and answer == value


# ---------------------------------------------------------------------------
# Call ids, cancellation and progress (Grasshopper UI plan, Task 1)
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """An in-process service on a free port, answering only the token ``secret``."""
    import threading

    from compas_acoustics.worker.service import WorkerService

    server = WorkerService(("127.0.0.1", 0), "secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield worker.Client(server.server_address[1], "secret")
    server.shutdown()
    server.server_close()


def test_jobs_record_progress_and_cancel():
    from compas_acoustics.worker import jobs

    jobs.progress("outside a job")  # a no-op without a job
    jobs.start("a")
    try:
        jobs.progress("x", 1, 3)
        assert jobs.status("a") == {"stage": "x", "done": 1, "of": 3, "cancelled": False}
        assert jobs.cancel("a") is True
        with pytest.raises(jobs.Cancelled):
            jobs.progress("y")
        with pytest.raises(jobs.Cancelled):
            jobs.check()
    finally:
        jobs.finish("a")
    assert jobs.status("a") is None
    assert jobs.cancel("a") is False


def test_a_call_id_gets_progress_and_is_forgotten_after_the_answer(service, monkeypatch):
    import threading
    import time

    from compas_acoustics.worker import api
    from compas_acoustics.worker import jobs

    release = threading.Event()

    def slow():
        jobs.progress("x", 1, 3)
        release.wait(5)
        return "done"

    monkeypatch.setattr(api, "info", slow)
    answer = {}
    thread = threading.Thread(target=lambda: answer.setdefault("result", service.call("info", call_id="job-1")))
    thread.start()
    for _ in range(100):
        if service.progress("job-1"):
            break
        time.sleep(0.02)
    assert service.progress("job-1") == {"stage": "x", "done": 1, "of": 3, "cancelled": False}
    release.set()
    thread.join(5)
    assert answer["result"] == "done"
    assert service.progress("job-1") is None


def test_a_cancelled_call_answers_499_and_stops_at_the_next_stage(service, monkeypatch):
    import threading
    import time

    from compas_acoustics.worker import api
    from compas_acoustics.worker import jobs

    reached = []

    def long():
        for i in range(500):
            jobs.progress("step", i, 500)
            reached.append(i)
            time.sleep(0.01)
        return "finished"

    monkeypatch.setattr(api, "info", long)
    errors = []

    def run():
        try:
            service.call("info", call_id="job-2")
        except worker.WorkerError as e:
            errors.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    while len(reached) < 5:
        time.sleep(0.01)
    assert service.cancel("job-2") is True
    thread.join(5)
    assert errors and errors[0].status == 499 and str(errors[0]) == "cancelled"
    assert len(reached) < 500


def test_cancel_of_an_unknown_id_is_false(service):
    assert service.cancel("nobody") is False


def test_progress_and_cancel_need_the_token(service):
    status, _ = worker._request(service.port, "/cancel/x", token="wrong")
    assert status == 401
    status, _ = worker._request(service.port, "/progress/x", body=None, token="wrong")
    assert status == 401


def test_a_call_without_an_id_runs_as_before(service):
    assert service.call("info")["compas_acoustics"] == compas_acoustics.__version__


def recorded_stages(monkeypatch):
    from compas_acoustics.worker import jobs

    stages = []
    monkeypatch.setattr(jobs, "progress", lambda stage, done=None, of=None: stages.append(stage))
    return stages


def test_the_hybrid_reports_its_stages(monkeypatch):
    pytest.importorskip("pyFDN")
    from compas_acoustics.simulation_methods import Ambisonics

    stages = recorded_stages(monkeypatch)
    scene().simulate(Ambisonics(tail="eyring", late_envelope=False))
    assert [s for s in dict.fromkeys(stages)] == ["references", "image sources", "rendering"]


def test_mono_reports_its_stages(monkeypatch):
    stages = recorded_stages(monkeypatch)
    scene().simulate(Mono(image_source_order=1, ray_tracing=False))
    assert list(dict.fromkeys(stages)) == ["image sources", "responses"]


def test_the_ray_field_reports_each_source(monkeypatch):
    from compas_acoustics import RayField

    stages = recorded_stages(monkeypatch)
    RayField.from_scene(scene(), n_rays=5, max_reflection_order=1)
    assert stages == ["tracing"]
