"""Run the parts of compas_acoustics that need Python >= 3.11 in a second interpreter.

Rhino 8 runs CPython 3.9, where the hybrid method and binaural decoding are not available. The worker is a local :mod:`compas.rpc`
server in another Python with compas_acoustics installed (see
:mod:`compas_acoustics.worker.install`); :func:`proxy` starts it on first use.
Scenes travel as COMPAS JSON, audio as WAV files in a temporary folder.

Every Python session starts its own worker, on a free port of 127.0.0.1, and shares a random token with it: calls
without the token are refused. The worker stops when the session ends, and when the session dies without saying so.
"""

import atexit
import contextlib
import hashlib
import importlib.util
import json
import os
import secrets
import socket
import tempfile
import uuid
from pathlib import Path

SERVICE = "compas_acoustics.worker.service"
# Pings, 0.1 s apart, before giving up on a starting worker; the first start of a fresh environment imports uncached
CONNECTION_ATTEMPTS = 1200
CONFIG = Path.home() / ".compas_acoustics" / "worker.json"
# One file per running worker, {"pid": ..., "port": ...}, so that the installer can stop them before replacing the environment
WORKERS = Path.home() / ".compas_acoustics" / "workers"
TOKEN_VARIABLE = "COMPAS_ACOUSTICS_WORKER_TOKEN"
PARENT_VARIABLE = "COMPAS_ACOUSTICS_WORKER_PARENT"
FOLDER_PREFIX = "compas_acoustics_worker_"

_PROXY = None

__all__ = ["CONFIG", "WORKERS", "worker_python", "fingerprint", "proxy", "stop", "stop_running_workers", "simulate", "convolve", "analyse"]


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


@contextlib.contextmanager
def _worker_environment(token):
    # PYTHONHOME and PYTHONPATH, inherited by the worker, would load this (Rhino's 3.9) Python's packages into the 3.12 worker
    saved = {name: os.environ.pop(name) for name in ("PYTHONHOME", "PYTHONPATH") if name in os.environ}
    os.environ[TOKEN_VARIABLE] = token
    os.environ[PARENT_VARIABLE] = str(os.getpid())
    try:
        yield
    finally:
        os.environ.pop(TOKEN_VARIABLE, None)
        os.environ.pop(PARENT_VARIABLE, None)
        os.environ.update(saved)


class _Worker:
    """A :class:`compas.rpc.Proxy` that sends the token with every call."""

    def __init__(self, proxy, token, port):
        self._proxy = proxy
        self._token = token
        self.port = port

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args):
            return getattr(self._proxy, name)(self._token, *args)

        return call

    def stop_server(self):
        self._proxy.stop_server()
        # The process is terminated, so it may not get to remove its own record
        for record in WORKERS.glob("*.json") if WORKERS.is_dir() else []:
            try:
                if json.loads(record.read_text())["port"] == self.port:
                    record.unlink(missing_ok=True)
            except (KeyError, OSError, ValueError):
                pass


def proxy():
    """The running worker, started on first use; None if no worker interpreter is configured."""
    global _PROXY
    if _PROXY is None:
        python = worker_python()
        if python is None:
            return None
        from compas.rpc import Proxy

        from compas_acoustics import __version__

        token = secrets.token_hex(16)
        port = _free_port()
        with _worker_environment(token):
            candidate = _Worker(
                Proxy(package="compas_acoustics.worker.api", python=python, port=port, service=SERVICE, autoreload=False,
                      max_conn_attempts=CONNECTION_ATTEMPTS),
                token, port)
        info = candidate.info()
        if (info["compas_acoustics"], info["fingerprint"]) != (__version__, fingerprint()):
            candidate.stop_server()
            raise RuntimeError(f"The worker at {python} runs a different compas_acoustics ({info['compas_acoustics']}) than this Python "
                               f"({__version__}): run the installer again.")
        atexit.register(stop)
        _PROXY = candidate
    return _PROXY


def stop():
    """Stop this session's worker, if it has one."""
    global _PROXY
    if _PROXY is not None:
        worker, _PROXY = _PROXY, None
        worker.stop_server()


def stop_running_workers(workers=None):
    """Ask every worker recorded in :data:`WORKERS` to shut down, and forget those that are gone.

    The installer calls this before it replaces the worker environment, whose files a running worker holds open on Windows.
    """
    import time
    from xmlrpc.client import ServerProxy

    folder = Path(WORKERS if workers is None else workers)
    for record in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            port = json.loads(record.read_text())["port"]
            ServerProxy(f"http://127.0.0.1:{port}").remote_shutdown()
        except Exception:
            pass
        for _ in range(100):
            if not record.exists():
                break
            time.sleep(0.1)
        record.unlink(missing_ok=True)


def _folder():
    return Path(tempfile.mkdtemp(prefix=FOLDER_PREFIX))


def _has_spharpy():
    return importlib.util.find_spec("spharpy") is not None


def simulate(scene, method, settings, references=None):
    """Run ``method`` (``"hybrid"`` or ``"pyroomacoustics"``) with ``settings`` on ``scene`` in the worker.

    The responses are stored on ``scene.ir_graph`` under ``method``, as :meth:`Scene.simulate` does,
    at the scene's ``sampling_rate``.

    Parameters
    ----------
    references : dict, optional
        ``{key: ImpulseResponse}`` passed to the method as ``reference``.

    Returns
    -------
    tuple of (dict, list of str)
        The responses by key, and the messages of the UserWarnings raised in the worker.
    """
    from compas_acoustics import ImpulseResponse
    from compas_acoustics.simulation_methods import SimulationContext

    folder = _folder()
    reference_paths = None
    if references:
        reference_paths = {key: str(ir.to_wav(folder / f"{uuid.uuid4().hex}.wav")[0]) for key, ir in references.items()}
    result = proxy().simulate(scene, method, settings, reference_paths, str(folder))
    pairs = {SimulationContext.key(s, r): (s, r) for s in scene.sources for r in scene.receivers}
    rirs = {}
    for key, path in result["responses"].items():
        source, receiver = pairs[key]
        # The WAV holds the peak-normalized samples and its JSON the raw level, but not which view the method returns
        rirs[key] = ImpulseResponse.from_wav(path, source=source, receiver=receiver, normalized=settings.get("normalize", True))
        scene.add_impulse_response(source, receiver, rirs[key], attribute=method)
    return rirs, result["warnings"]


def convolve(auraliser):
    """``auraliser.convolve()``, in the worker when binaural decoding needs spharpy and this Python has none.

    Returns the signal before any output gain, as :meth:`Auraliser.convolve` does.
    """
    if auraliser.output_format != "binaural" or _has_spharpy() or worker_python() is None:
        return auraliser.convolve()
    import numpy as np
    import soundfile as sf

    folder = _folder()
    impulse_responses = {}
    for ir_id, data in auraliser.impulse_responses.items():
        path = folder / f"{uuid.uuid4().hex}.wav"
        sf.write(str(path), np.asarray(data["ir"], dtype=float), int(data["sampling_rate"]), subtype="FLOAT")
        impulse_responses[ir_id] = {key: value for key, value in data.items() if key != "ir"}
        impulse_responses[ir_id]["path"] = str(path)
    sounds = {key: sound["filepath"] for key, sound in auraliser.anechoic_sounds.items()}
    out = proxy().convolve(auraliser.name, impulse_responses, sounds, auraliser.output_format,
                           str(auraliser.hrtf_path), int(auraliser.sampling_rate), str(folder / "convolved.wav"))
    signal, _ = sf.read(out, dtype="float64", always_2d=True)
    return signal


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

    analyser = AcousticAnalyser(ir)
    if _pyrato_has_parameters() or worker_python() is None or not hasattr(ir, "to_wav"):
        analyser.analyse()
        return analyser, []
    with warnings.catch_warnings():
        # this pyrato cannot compute them; the worker replaces the NaNs
        warnings.filterwarnings("ignore", message="STI is not available")
        analyser.analyse()
    path, _ = ir.to_wav(_folder() / f"{uuid.uuid4().hex}.wav")
    result = proxy().analyse(str(path))
    for name in NEEDS_PYRATO_1:
        analyser.analysis_data[name] = result["analysis"][name]
        analyser.analysis_data[f"{name}_overall"] = result["analysis"][f"{name}_overall"]
    return analyser, result["warnings"]
