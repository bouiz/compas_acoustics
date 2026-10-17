"""Functions the worker serves; called through :func:`compas_acoustics.worker.proxy`, never directly.

Arguments and results are COMPAS JSON: scenes as objects, audio as WAV file paths.
"""

import importlib.util
import sys
import uuid
import warnings
from pathlib import Path


def info():
    from compas_acoustics import __version__

    return {"python": sys.version.split()[0], "compas_acoustics": __version__, "hybrid": importlib.util.find_spec("pyFDN") is not None}


def simulate(scene, method, settings, sampling_rate, references, folder):
    from compas_acoustics import ImpulseResponse
    from compas_acoustics.methods import Pyroomacoustics
    from compas_acoustics.methods import SimulationContext

    if method == "hybrid":
        from compas_acoustics.methods import Hybrid as method_class
    elif method == "pyroomacoustics":
        method_class = Pyroomacoustics
    else:
        raise ValueError(f"Unknown method {method!r}: 'hybrid' or 'pyroomacoustics'.")
    settings = dict(settings)
    if references:
        pairs = {SimulationContext.key(s, r): (s, r) for s in scene.sources for r in scene.receivers}
        settings["reference"] = {key: ImpulseResponse.from_wav(path, source=pairs[key][0], receiver=pairs[key][1]) for key, path in references.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rirs = scene.simulate(method_class(**settings), sampling_rate=sampling_rate)
    folder = Path(folder)
    responses = {key: str(ir.to_wav(folder / f"{uuid.uuid4().hex}.wav")[0]) for key, ir in rirs.items()}
    return {"responses": responses, "warnings": [str(w.message) for w in caught if issubclass(w.category, UserWarning)]}


def analyse(path):
    from compas_acoustics import AcousticAnalyser
    from compas_acoustics import ImpulseResponse

    return AcousticAnalyser(ImpulseResponse.from_wav(path)).analyse()


def convolve(name, impulse_responses, anechoic_sounds, output_format, hrtf_path, sample_rate, out_path):
    import soundfile as sf

    from compas_acoustics import Auraliser

    irs = {}
    for ir_id, data in impulse_responses.items():
        data = dict(data)
        ir, _ = sf.read(data.pop("path"), dtype="float64", always_2d=True)
        data["ir"] = ir
        irs[ir_id] = data
    sounds = {key: {"filepath": path} for key, path in anechoic_sounds.items()}
    auraliser = Auraliser(name, irs, sounds, output_format=output_format, sample_rate=sample_rate)
    auraliser.hrtf_path = hrtf_path
    sf.write(out_path, auraliser.convolve(), int(sample_rate), subtype="FLOAT")
    return out_path
