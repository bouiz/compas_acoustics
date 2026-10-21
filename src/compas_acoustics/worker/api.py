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

__all__ = ["info", "simulate", "analyse"]


def _worker_folder(path):
    """``path`` resolved, if it is one of the temporary folders :func:`compas_acoustics.worker.simulate` and ``analyse`` make."""
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
        method = method_class(**settings)
        rirs = scene.simulate(method)
    formats = {method.attribute: rirs, **method.extra_responses}
    responses = {attribute: {key: str(ir.to_wav(folder / f"{uuid.uuid4().hex}.wav")[0]) for key, ir in irs.items()}
                 for attribute, irs in formats.items()}
    return {"responses": responses, "warnings": [str(w.message) for w in caught if issubclass(w.category, UserWarning)]}


def analyse(path):
    from compas_acoustics import AcousticAnalyser
    from compas_acoustics import ImpulseResponse

    # from_wav restores the raw level and the generation, which G needs
    ir = ImpulseResponse.from_wav(_worker_file(path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analysis = AcousticAnalyser(ir).analyse()
    return {"analysis": analysis, "warnings": [str(w.message) for w in caught if issubclass(w.category, UserWarning)]}
