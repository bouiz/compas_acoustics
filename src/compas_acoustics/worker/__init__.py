"""Run the parts of compas_acoustics that need Python >= 3.11 in a second interpreter.

Rhino 8 runs CPython 3.9, where the hybrid method, STI, sound strength G and
binaural decoding are not available. The worker is a local :mod:`compas.rpc`
server in another Python with compas_acoustics installed (see
:mod:`compas_acoustics.worker.install`); :func:`proxy` starts it on first use.
Scenes travel as COMPAS JSON, audio as WAV files in a temporary folder.
"""

import contextlib
import importlib.util
import json
import os
import tempfile
import uuid
import warnings
from pathlib import Path

PORT = 17531
SERVICE = "compas_acoustics.worker.service"
# Pings, 0.1 s apart, before giving up on a starting worker; the first start of a fresh environment imports uncached
CONNECTION_ATTEMPTS = 1200
CONFIG = Path.home() / ".compas_acoustics" / "worker.json"

_PROXY = None

__all__ = ["PORT", "CONFIG", "worker_python", "proxy", "simulate", "analyse", "convolve"]


def worker_python():
    """The worker's interpreter: ``$COMPAS_ACOUSTICS_WORKER_PYTHON``, else the one in :data:`CONFIG`; None if neither exists."""
    python = os.environ.get("COMPAS_ACOUSTICS_WORKER_PYTHON")
    if not python and CONFIG.exists():
        python = json.loads(CONFIG.read_text()).get("python")
    return python if python and os.path.exists(python) else None


@contextlib.contextmanager
def _without_python_path():
    # Inherited by the worker, these would load this (Rhino's 3.9) Python's packages into the 3.12 worker
    saved = {name: os.environ.pop(name) for name in ("PYTHONHOME", "PYTHONPATH") if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def proxy():
    """The running worker, started on first use; None if no worker interpreter is configured."""
    global _PROXY
    if _PROXY is None:
        python = worker_python()
        if python is None:
            return None
        from compas.rpc import Proxy

        from compas_acoustics import __version__

        with _without_python_path():
            candidate = Proxy(package="compas_acoustics.worker.api", python=python, port=PORT, service=SERVICE, autoreload=False,
                              max_conn_attempts=CONNECTION_ATTEMPTS)
        version = candidate.info()["compas_acoustics"]
        if version != __version__:
            candidate.stop_server()
            raise RuntimeError(f"The worker at {python} has compas_acoustics {version}, this Python has {__version__}: run the installer again.")
        _PROXY = candidate
    return _PROXY


def _folder():
    return Path(tempfile.mkdtemp(prefix="compas_acoustics_worker_"))


def _has_spharpy():
    return importlib.util.find_spec("spharpy") is not None


def simulate(scene, method, settings, sampling_rate, references=None):
    """Run ``method`` (``"hybrid"`` or ``"pyroomacoustics"``) with ``settings`` on ``scene`` in the worker.

    The responses are stored on ``scene.ir_graph`` under ``method``, as :meth:`Scene.simulate` does.

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
    result = proxy().simulate(scene, method, settings, sampling_rate, reference_paths, str(folder))
    pairs = {SimulationContext.key(s, r): (s, r) for s in scene.sources for r in scene.receivers}
    rirs = {}
    for key, path in result["responses"].items():
        source, receiver = pairs[key]
        rirs[key] = ImpulseResponse.from_wav(path, source=source, receiver=receiver)
        scene.add_impulse_response(source, receiver, rirs[key], attribute=method)
    return rirs, result["warnings"]


def analyse(ir):
    """An :class:`AcousticAnalyser` of ``ir`` after :meth:`~AcousticAnalyser.analyse`.

    Analysed in the worker when this Python's pyrato cannot compute STI and G
    and a worker is configured; here otherwise, or if the worker fails (with a warning).
    """
    from compas_acoustics import AcousticAnalyser
    from compas_acoustics.analyser import _STI_SUPPORTED

    analyser = AcousticAnalyser(ir)
    if _STI_SUPPORTED or worker_python() is None or not hasattr(ir, "to_wav"):
        analyser.analyse()
        return analyser
    try:
        path = ir.to_wav(_folder() / "ir.wav")[0]
        analyser.analysis_data = proxy().analyse(str(path))
    except Exception as error:
        warnings.warn(f"The worker failed, analysing here (STI and G are NaN): {error}")
        analyser.analyse()
    return analyser


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
        sf.write(str(path), np.asarray(data["ir"], dtype=float), int(data["sample_rate"]), subtype="FLOAT")
        impulse_responses[ir_id] = {key: value for key, value in data.items() if key != "ir"}
        impulse_responses[ir_id]["path"] = str(path)
    sounds = {key: sound["filepath"] for key, sound in auraliser.anechoic_sounds.items()}
    out = proxy().convolve(auraliser.name, impulse_responses, sounds, auraliser.output_format,
                           str(auraliser.hrtf_path), int(auraliser.sample_rate), str(folder / "convolved.wav"))
    signal, _ = sf.read(out, dtype="float64", always_2d=True)
    return signal
