"""Functions the worker serves; called through :func:`compas_acoustics.worker.proxy`, never directly.

Arguments and results are COMPAS JSON: scenes as objects, audio as WAV file paths.
The worker reads and writes the responses only inside the temporary folders its client made for them.
"""

import importlib.util
import sys
import tempfile
import uuid
import warnings
from pathlib import Path

from compas_acoustics.worker import FOLDER_PREFIX

__all__ = ["info", "simulate", "convolve"]


def _worker_folder(path):
    """``path`` resolved, if it is one of the temporary folders :func:`compas_acoustics.worker.simulate` and ``convolve`` make."""
    path = Path(path).resolve()
    if path.parent != Path(tempfile.gettempdir()).resolve() or not path.name.startswith(FOLDER_PREFIX):
        raise ValueError(f"{path} is not a compas_acoustics worker folder.")
    return path


def _worker_file(path):
    path = Path(path)
    return _worker_folder(path.parent) / path.name


def info():
    from compas_acoustics import __version__
    from compas_acoustics.worker import fingerprint

    return {"python": sys.version.split()[0], "compas_acoustics": __version__, "fingerprint": fingerprint(),
            "hybrid": importlib.util.find_spec("pyFDN") is not None}


def simulate(scene, method, settings, references, folder):
    from compas_acoustics import ImpulseResponse
    from compas_acoustics.simulation_methods import Pyroomacoustics
    from compas_acoustics.simulation_methods import SimulationContext

    if method == "hybrid":
        from compas_acoustics.simulation_methods import Hybrid as method_class
    elif method == "pyroomacoustics":
        method_class = Pyroomacoustics
    else:
        raise ValueError(f"Unknown method {method!r}: 'hybrid' or 'pyroomacoustics'.")
    folder = _worker_folder(folder)
    settings = dict(settings)
    if references:
        pairs = {SimulationContext.key(s, r): (s, r) for s in scene.sources for r in scene.receivers}
        settings["reference"] = {key: ImpulseResponse.from_wav(_worker_file(path), source=pairs[key][0], receiver=pairs[key][1])
                                 for key, path in references.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rirs = scene.simulate(method_class(**settings))
    responses = {key: str(ir.to_wav(folder / f"{uuid.uuid4().hex}.wav")[0]) for key, ir in rirs.items()}
    return {"responses": responses, "warnings": [str(w.message) for w in caught if issubclass(w.category, UserWarning)]}


def convolve(name, impulse_responses, anechoic_sounds, output_format, hrtf_path, sampling_rate, out_path):
    import soundfile as sf

    from compas_acoustics import Auraliser

    out_path = _worker_file(out_path)
    irs = {}
    for ir_id, data in impulse_responses.items():
        data = dict(data)
        ir, _ = sf.read(str(_worker_file(data.pop("path"))), dtype="float64", always_2d=True)
        data["ir"] = ir
        irs[ir_id] = data
    sounds = {key: {"filepath": path} for key, path in anechoic_sounds.items()}
    auraliser = Auraliser(name, irs, sounds, output_format=output_format, sampling_rate=sampling_rate)
    auraliser.hrtf_path = hrtf_path
    sf.write(str(out_path), auraliser.convolve(), int(sampling_rate), subtype="FLOAT")
    return str(out_path)
