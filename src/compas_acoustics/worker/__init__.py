"""Run compas_acoustics in a second interpreter, the worker, and call it over HTTP with compas_pb payloads.

Rhino 8 runs CPython 3.9, where the binaural and ambisonics methods and pyrato's STI and G are not available. The worker is a local
HTTP service (:mod:`compas_acoustics.worker.service`) in another Python with compas_acoustics installed (see
:mod:`compas_acoustics.worker.install`); :func:`client` starts it on first use, and the Grasshopper plugin starts its own.
Arguments and results travel as compas_pb messages (:mod:`compas_acoustics.pb`): scenes, responses and audio included.

Every session starts its own worker, on a free port of 127.0.0.1, and shares a random token with it: calls
without the token are refused. The worker stops when the session ends, and when the session dies without saying so.
"""

import atexit
import hashlib
import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
from pathlib import Path

SERVICE = "compas_acoustics.worker.service"
# Pings, PING_SECONDS apart, before giving up on a starting worker; the first start of a fresh environment imports uncached
CONNECTION_ATTEMPTS = 1200
PING_SECONDS = 0.1
CONFIG = Path.home() / ".compas_acoustics" / "worker.json"
# One file per running worker, {"pid", "port", "token"}, so that the installer can stop them before replacing the environment
WORKERS = Path.home() / ".compas_acoustics" / "workers"
TOKEN_VARIABLE = "COMPAS_ACOUSTICS_WORKER_TOKEN"
PARENT_VARIABLE = "COMPAS_ACOUSTICS_WORKER_PARENT"
CONTENT_TYPE = "application/x-protobuf"

_CLIENT = None

__all__ = [
    "CONFIG",
    "WORKERS",
    "Client",
    "WorkerError",
    "worker_python",
    "fingerprint",
    "client",
    "stop",
    "stop_running_workers",
    "simulate",
    "analyse",
]


class WorkerError(RuntimeError):
    """An error the worker answered with: ``status`` is the HTTP status, the message the worker's text."""

    def __init__(self, status, text):
        super().__init__(text)
        self.status = status


def worker_python():
    """The worker's interpreter: ``$COMPAS_ACOUSTICS_WORKER_PYTHON``, else the one in :data:`CONFIG`; None if neither exists."""
    python = os.environ.get("COMPAS_ACOUSTICS_WORKER_PYTHON")
    if not python and CONFIG.exists():
        python = json.loads(CONFIG.read_text()).get("python")
    return python if python and os.path.exists(python) else None


def fingerprint():
    """A hash of the package's Python sources, which tells whether two installations run the same code.

    The version number does not: it stays the same between reinstalls of a development version.
    """
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _worker_environment(token):
    # PYTHONHOME and PYTHONPATH would load this (Rhino's 3.9) Python's packages into the 3.12 worker
    env = {name: value for name, value in os.environ.items() if name not in ("PYTHONHOME", "PYTHONPATH")}
    env[TOKEN_VARIABLE] = token
    env[PARENT_VARIABLE] = str(os.getpid())
    return env


def _spawn(python, port, env, log):
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}  # CREATE_NO_WINDOW
    return subprocess.Popen([python, "-m", SERVICE, "--port", str(port)], env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, **kwargs)


def _request(port, path, body=b"", token=None, timeout=None, headers=None):
    """POST ``body`` to the worker, or GET with ``body=None`` (and for ``/ping``): ``(status, body)``.

    An error status is an answer, not an exception.
    """
    import urllib.error
    import urllib.request

    headers = dict(headers or {}, **{"Content-Type": CONTENT_TYPE})
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    method = "GET" if path == "/ping" or body is None else "POST"
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if method == "GET" else body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, answer.read()
    except urllib.error.HTTPError as answer:
        return answer.code, answer.read()


class Client:
    """A running worker: ``client.simulate(...)`` calls :func:`compas_acoustics.worker.api.simulate` there."""

    def __init__(self, port, token, process=None, python=None, log=None):
        self.port = port
        self._token = token
        self._process = process
        self.python = python
        self._log = log

    def _send(self, path, body=b"", authorize=True, headers=None):
        return _request(self.port, path, body, self._token if authorize else None,
                        timeout=None if path.startswith("/api/") else 2, headers=headers)

    def ping(self):
        try:
            return self._send("/ping", authorize=False)[0] == 200
        except OSError:
            return False

    def call(self, function, *args, call_id=None, **kwargs):
        """Call ``function`` of the worker API; with a ``call_id``, :meth:`cancel` and :meth:`progress` reach the call."""
        from compas_pb import pb_dump_bts

        import compas_acoustics.pb  # noqa: F401  (registers the converters)
        from compas_acoustics.pb import plain

        headers = {"X-Call-Id": call_id} if call_id else None
        return self._result(*self._send(f"/api/{function}", pb_dump_bts({"args": plain(list(args)), "kwargs": plain(kwargs)}), headers=headers))

    def _result(self, status, body):
        from compas_pb import pb_load_bts

        answer = pb_load_bts(body) if body else {}
        if status != 200:
            raise WorkerError(status, answer.get("error", f"HTTP {status}") if isinstance(answer, dict) else f"HTTP {status}")
        return answer["result"]

    def cancel(self, call_id):
        """Ask the call ``call_id`` to stop at its next stage; False if the worker has no such call."""
        return self._result(*self._send(f"/cancel/{call_id}"))

    def progress(self, call_id):
        """The last stage the call ``call_id`` reported, ``{"stage", "done", "of", "cancelled"}``; None once it answered."""
        status, body = self._send(f"/progress/{call_id}", body=None)
        return None if status == 404 else self._result(status, body)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.call(name, *args, **kwargs)

    def _log_tail(self):
        try:
            return Path(self._log.name).read_text(errors="replace")[-2000:]
        except (AttributeError, OSError):
            return ""

    def stop(self):
        """Ask the worker to shut down, and end its process if it does not."""
        try:
            self._send("/shutdown")
        except OSError:
            pass
        if self._process is not None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        # The process may be killed before it removes its own record
        _forget(lambda record: record.get("port") == self.port)

    @classmethod
    def start(cls, python):
        """Start a worker with ``python`` and wait until it answers; refuse one that runs other compas_acoustics sources."""
        from compas_acoustics import __version__

        token = secrets.token_hex(16)
        port = _free_port()
        log = tempfile.NamedTemporaryFile(prefix="compas_acoustics_worker_", suffix=".log", delete=False)
        candidate = cls(port, token, _spawn(python, port, _worker_environment(token), log), python, log)
        for _ in range(CONNECTION_ATTEMPTS):
            if candidate.ping():
                break
            if candidate._process.poll() is not None:
                raise RuntimeError(f"The worker at {python} exited while starting:\n{candidate._log_tail()}")
            time.sleep(PING_SECONDS)
        else:
            candidate.stop()
            raise RuntimeError(f"The worker at {python} did not answer within {CONNECTION_ATTEMPTS * 0.1:.0f} s.")
        info = candidate.info()
        if (info["compas_acoustics"], info["fingerprint"]) != (__version__, fingerprint()):
            candidate.stop()
            raise RuntimeError(f"The worker at {python} runs a different compas_acoustics ({info['compas_acoustics']}) than this Python "
                               f"({__version__}): run the installer again.")
        return candidate


def _forget(matches):
    for record in sorted(WORKERS.glob("*.json")) if WORKERS.is_dir() else []:
        try:
            if matches(json.loads(record.read_text())):
                record.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def client():
    """This session's worker, started on first use; None if no worker interpreter is configured."""
    global _CLIENT
    if _CLIENT is None:
        python = worker_python()
        if python is None:
            return None
        _CLIENT = Client.start(python)
        atexit.register(stop)
    return _CLIENT


def stop():
    """Stop this session's worker, if it has one."""
    global _CLIENT
    if _CLIENT is not None:
        running, _CLIENT = _CLIENT, None
        running.stop()


def stop_running_workers(workers=None):
    """Ask every worker recorded in :data:`WORKERS` to shut down, and forget those that are gone.

    The installer calls this before it replaces the worker environment, whose files a running worker holds open on Windows.
    """
    folder = Path(WORKERS if workers is None else workers)
    for record in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            entry = json.loads(record.read_text())
            _request(entry["port"], "/shutdown", token=entry.get("token", ""), timeout=2)
        except Exception:
            pass
        for _ in range(100):
            if not record.exists():
                break
            time.sleep(0.1)
        record.unlink(missing_ok=True)


def simulate(scene, method, settings, references=None):
    """Run ``method`` (``"binaural"``, ``"ambisonics"`` or ``"mono"``) with ``settings`` on ``scene`` in the worker.

    The responses are stored on ``scene.ir_graph`` as :meth:`Scene.simulate` stores them, at the scene's ``sampling_rate``,
    under the method's name.

    Parameters
    ----------
    references : dict, optional
        ``{key: ImpulseResponse}`` passed to the method as ``reference``.

    Returns
    -------
    tuple of (dict, list of str)
        The responses by key, as :meth:`Scene.simulate` returns them, and the messages of the UserWarnings
        raised in the worker.
    """
    from compas_acoustics.simulation_methods import SimulationContext

    result = client().simulate(scene, method, settings, references or None)
    pairs = {SimulationContext.key(s, r): (s, r) for s in scene.sources for r in scene.receivers}
    ((attribute, responses),) = result["responses"].items()
    for key, ir in responses.items():
        # the worker's copies of the source and receiver are replaced by the scene's own
        ir.source, ir.receiver = pairs[key]
        scene.add_impulse_response(ir.source, ir.receiver, ir, attribute=attribute)
    return responses, result["warnings"]


#: The parameters that need pyrato >= 1.0, which Rhino 8's Python 3.9 cannot install
NEEDS_PYRATO_1 = ("STI", "G")


def _pyrato_has_parameters():
    from compas_acoustics import analyser

    return analyser._STI_SUPPORTED


def analyse(ir):
    """Analyse ``ir`` here, with STI and G from the worker when this Python's pyrato cannot compute them.

    STI and G need pyrato >= 1.0, which needs Python >= 3.11. With a worker configured they are computed
    there; every other parameter is computed here, as without a worker.

    Parameters
    ----------
    ir : :class:`compas_acoustics.ImpulseResponse`

    Returns
    -------
    tuple of (:class:`compas_acoustics.AcousticAnalyser`, list of str)
        The analyser, its :attr:`analysis_data` filled, and the messages of the UserWarnings raised in the worker.
    """
    import warnings

    from compas_acoustics import AcousticAnalyser
    from compas_acoustics import ImpulseResponse

    analyser = AcousticAnalyser(ir)
    if _pyrato_has_parameters() or worker_python() is None or not isinstance(ir, ImpulseResponse):
        analyser.analyse()
        return analyser, []
    with warnings.catch_warnings():
        # this pyrato cannot compute them; the worker replaces the NaNs
        warnings.filterwarnings("ignore", message="STI is not available")
        analyser.analyse()
    result = client().analyse(ir)
    for name in NEEDS_PYRATO_1:
        analyser.analysis_data[name] = result["analysis"][name]
        analyser.analysis_data[f"{name}_overall"] = result["analysis"][f"{name}_overall"]
    return analyser, result["warnings"]
