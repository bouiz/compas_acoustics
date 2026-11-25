"""Functions the worker serves, called through :func:`compas_acoustics.worker.client` or the Grasshopper plugin.

Arguments and results are what :mod:`compas_acoustics.pb` carries: Data objects, dicts, lists and numbers. Unusable
input raises ValueError or TypeError, which the service answers with its message.
"""

import functools
import importlib.util
import math
import os
import random
import warnings
from contextlib import contextmanager

import numpy as np
import soundfile as sf
from compas.datastructures import Mesh
from compas.geometry import Polyline
from compas.geometry import Vector

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import Analyser
from compas_acoustics import AudioPlayer
from compas_acoustics import Defaults
from compas_acoustics import EnergyMap
from compas_acoustics import MaterialsLibrary
from compas_acoustics import RayField
from compas_acoustics import Receiver
from compas_acoustics import RoomGeometry
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics import SpecularReflections
from compas_acoustics import __version__
from compas_acoustics import simulation_methods
from compas_acoustics.analyser import PARAMETERS
from compas_acoustics.analyser import frequency_response
from compas_acoustics.analyser import time_response
from compas_acoustics.auraliser import Auraliser
from compas_acoustics.fitness import regularized_fitness as _regularized_fitness
from compas_acoustics.pb import SCHEMA
from compas_acoustics.pb.types import Audio
from compas_acoustics.room.pra import CARDIOID_P
from compas_acoustics.worker import fingerprint
from compas_acoustics.worker import jobs

__all__ = [
    "info",
    "simulate",
    "analyse",
    "materials_library",
    "material_from_name",
    "make_material",
    "make_source",
    "make_receiver",
    "make_acoustic_object",
    "make_scene",
    "room_measures",
    "analysis",
    "auralise",
    "save_impulse_responses",
    "play",
    "stop",
    "ray_field",
    "energy_map",
    "specular_reflections",
    "regularized_fitness",
    "pause",
    "resume",
    "playback_status",
    "sound_samples",
    "sound_info",
]


def _method_class(name):
    if name in ("binaural", "ambisonics", "mono"):
        return getattr(simulation_methods, name.capitalize())
    raise ValueError(f"Unknown method {name!r}: 'binaural', 'ambisonics' or 'mono'.")


@contextmanager
def _user_warnings():
    """Record the block's warnings; the list it yields holds the distinct UserWarning messages once the block ends."""
    messages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield messages
    messages.extend(dict.fromkeys(str(w.message) for w in caught if issubclass(w.category, UserWarning)))


def info():
    return {
        "compas_acoustics": __version__,
        "fingerprint": fingerprint(),
        "schema": SCHEMA,
        "speed_of_sound": Defaults().DEFAULT_SPEED_OF_SOUND,
        "hybrid": importlib.util.find_spec("pyFDN") is not None,
    }


def simulate(scene, method, settings, references=None):
    """``{"responses": {method name: {key: ImpulseResponse}}, "warnings": [str]}`` of ``scene`` simulated.

    ``method`` is ``'binaural'``, ``'ambisonics'`` or ``'mono'``, ``settings`` its keyword arguments and ``references``
    its ``reference``, keyed as :meth:`SimulationContext.key` keys a pair.
    """
    method_class = _method_class(method)
    settings = dict(settings or {})
    if references:
        settings["reference"] = dict(references)
    with _user_warnings() as messages:
        method = method_class(**settings)
        rirs = scene.simulate(method)
    return {"responses": {method.name: dict(rirs)}, "warnings": messages}


def analyse(ir):
    """STI and G of ``ir`` for a client whose pyrato cannot compute them: ``{"analysis": {...}, "warnings": [str]}``."""
    with _user_warnings() as messages:
        # the client computes every other parameter itself
        result = Analyser(ir)._sti_and_strength()
    return {"analysis": result, "warnings": messages}


# =============================================================================
# Materials and the model: what the Material, Source, Receiver, Acoustic Object and Acoustic Scene components make
# =============================================================================


@functools.lru_cache(maxsize=None)
def _library():
    return MaterialsLibrary()


def materials_library():
    """The library's names by category, in its category order: ``[[category, [name, ...]], ...]``."""
    return [[category.value, list(names)] for category, names in _library().materials_by_category.items()]


def material_from_name(name):
    library = _library()
    if name not in library.material_names:
        raise ValueError(f"No material {name!r} in the materials library.")
    return library.material_from_name(name)


def make_material(name, absorption, scattering=None, category=None):
    """An :class:`AcousticMaterial`; an empty ``scattering`` or ``category`` is left to the class's default."""
    return AcousticMaterial(name=name, absorption_coefficients=list(absorption), scattering_coefficients=list(scattering) if scattering else None, category=category or None)


def make_source(name, position, direction=None, directivity=None, sound_file=None, spl=None):
    """A :class:`Source` with the defaults of the Source component: ``my_source``, facing +x, omnidirectional.

    ``spl``: one value for all bands or one per band; None or empty keeps the default.
    """
    directivity = directivity or "omnidirectional"
    if directivity not in CARDIOID_P:
        raise ValueError("Input parameter 'directivity' is not valid.")
    source = Source(
        name or "my_source",
        *(position if position is not None else [0, 0, 0]),
        directivity=directivity,
        direction=list(direction) if direction is not None else [1, 0, 0],
        sound_file=sound_file or None,
    )
    if spl is not None and len(list(spl)) > 0:
        source.spl = spl
    return source


def make_receiver(name, position=None, direction=None):
    """``{"receiver": Receiver}``, upright (+z up), as the Receiver component makes it."""
    receiver = Receiver(name or "my_receiver", *(position if position is not None else [0, 0, 1]))
    if direction is not None:
        receiver.direction = Vector(*direction)
    # raises for a receiver facing straight up or down, whose orientation is undefined
    receiver.frame
    return {"receiver": receiver}


def make_acoustic_object(geometry, material, name=None, role="boundary"):
    """An :class:`AcousticObject` of a Mesh or a Polyline."""
    if not isinstance(geometry, (Mesh, Polyline)):
        raise TypeError("Unsupported geometry type.")
    return AcousticObject(acoustic_geometry=geometry, material=material, name=name or None, role=role or "boundary")


def make_scene(objects, sources, receivers, name=None, sampling_rate=None):
    """A :class:`Scene` of the objects, sources and receivers; an item of the wrong type is a TypeError naming it."""
    for label, items, kind in (("objects", objects, AcousticObject), ("sources", sources, Source), ("receivers", receivers, Receiver)):
        for i, item in enumerate(items):
            if not isinstance(item, kind):
                raise TypeError(f"{label}[{i}] is a {type(item).__name__}, not a {kind.__name__}.")
    scene = Scene(name=name or None, sampling_rate=sampling_rate)
    scene.add_acoustic_objects(list(objects))
    for source in sources:
        scene.add_source(source)
    for receiver in receivers:
        scene.add_receiver(receiver)
    return scene


def room_measures(scene):
    """``{"volume", "area"}`` of ``scene`` as the simulation methods take them, in m3 and m2: the volume its surfaces
    close (None if they close none) and the area exposed to the air, the ``S`` of Sabine and Eyring (None without
    surfaces)."""
    try:
        geometry = RoomGeometry.from_scene(scene)
    except ValueError:
        # a scene without surfaces
        return {"volume": None, "area": None}
    return {"volume": float(geometry.volume) if geometry.is_closed else None, "area": geometry.surface_area}


# =============================================================================
# Analysis: what Analyse IR shows
# =============================================================================

TIME_SCALES = ["db", "linear"]


def _parameter(name, title, x_unit, y_unit, x_scale, x, values, overall=None, channels=()):
    return {
        "name": name,
        "title": title,
        "x_unit": x_unit,
        "y_unit": y_unit,
        "x_scale": x_scale,
        "frequencies": [float(v) for v in x],
        "values": [[float(v) for v in row] for row in values],
        "overall": None if overall is None else float(overall),
        "channels": list(channels),
    }


def _plotted_channels(ir):
    """The raw channels a curve shows, (samples, channels), and their names: W of an ambisonic response, both ears of
    a binaural one."""
    data = ir.denormalized()
    data = data[:, np.newaxis] if data.ndim == 1 else data
    if ir.format == "binaural":
        return data, ["left", "right"]
    if data.shape[1] > 1:
        return data[:, :1], ["W"]
    return data, []


def _curve_title(title, channels):
    return "{} ({})".format(title, ", ".join(channels)) if channels else title


def analysis(ir, time_scale="db"):
    """``{"parameters": [parameter], "time": parameter, "frequency": parameter, "warnings": [str]}`` of ``ir``, as
    Analyse IR shows them.

    A parameter is ``{"name", "title", "x_unit", "y_unit", "x_scale" ('bands', 'linear' or 'log'), "frequencies",
    "values" (one list per channel), "overall", "channels"}``; STI, G, SPL, LF and LFC are left out where all their
    values are NaN. ``time_scale`` ``'db'`` gives the time response in dB re its peak, ``'linear'`` its amplitude to
    60 dB down its decay.
    """
    time_scale = time_scale or TIME_SCALES[0]
    if time_scale not in TIME_SCALES:
        raise ValueError(f"Input parameter 'time_scale' must be one of {TIME_SCALES}.")
    with _user_warnings() as messages:
        analyser = Analyser(ir)
        analyser.analyse()
        data = analyser.analysis_data
        bands = analyser.frequency_bands
        parameters = []
        for name, (title, x_unit, y_unit) in PARAMETERS.items():
            values = [float(v) for v in data[name]]
            if name in ("STI", "G", "SPL", "LF", "LFC") and all(math.isnan(v) for v in values):
                continue
            parameters.append(_parameter(name, title, x_unit, y_unit, "bands", bands, [values], data[f"{name}_overall"]))
        samples, channels = _plotted_channels(ir)
        db = time_scale == "db"
        times, levels = time_response(samples, ir.sampling_rate, db=db, trim_db=None if db else 60.0)
        time = _parameter("IR", _curve_title("Impulse Response", channels), "Time (s)", "Level (dB)" if db else "Amplitude", "linear", times, levels, channels=channels)
        frequencies, magnitudes = frequency_response(samples, ir.sampling_rate, num_fractions=3, frequency_range=(20.0, 20000.0))
        frequency = _parameter("FR", _curve_title("Frequency Response, 1/3 octave", channels), "Frequency (Hz)", "Level (dB)", "log", frequencies, magnitudes, channels=channels)
    return {"parameters": parameters, "time": time, "frequency": frequency, "warnings": messages}


# =============================================================================
# Auralisation and playback: what Auralise and the Audio Player do
# =============================================================================

LEVELS = ["relative", "individual", "none"]


def _checked_level(level):
    level = level or LEVELS[0]
    if level not in LEVELS:
        raise ValueError(f"Input parameter 'level' must be one of {LEVELS}.")
    return level


def _receivers(group):
    """The ``(ir, sound)`` pairs of one ``{"branch", "irs", "sounds"}`` group per receiver, in order of appearance; a
    None sound is the sound file of the response's source."""
    irs = list(group["irs"])
    sounds = list(group.get("sounds") or [None] * len(irs))
    if len(sounds) != len(irs):
        raise ValueError("DataTree branch length mismatch.")
    receivers = {}
    for ir, sound in zip(irs, sounds):
        receivers.setdefault(str(ir.receiver.guid), []).append((ir, sound or ir.source.sound_file))
    return list(receivers.values())


def auralise(groups, level="relative", mix_sources=False, save_folder=None, save_name=None):
    """``{"signals": [{"branch", "audio": Audio}], "peak_dbfs": float, "warnings": [str]}``: the impulse responses
    convolved with anechoic sounds per receiver, as Auralise does.

    ``groups`` holds one ``{"branch", "irs", "sounds"}`` per Grasshopper branch; a None sound is the sound file of the
    response's source. One signal per source and receiver, or per receiver with ``mix_sources``. ``level``:
    ``'relative'`` (one gain for the call, the loudest at -1 dBFS), ``'individual'`` or ``'none'``. With a
    ``save_folder`` each signal is also written there as ``<save_name or 'convolved_sound'>_mic_<suffix>.wav``.
    """
    level = _checked_level(level)
    renders = []
    with _user_warnings() as messages:
        for group in groups:
            for r, pairs in enumerate(_receivers(group)):
                tasks = [(pairs, f"{r}")] if mix_sources else [([pair], f"{r}_{j}") for j, pair in enumerate(pairs)]
                for task, suffix in tasks:
                    jobs.progress("rendering", len(renders))
                    task_formats = sorted({ir.format for ir, _ in task})
                    if len(task_formats) > 1:
                        raise ValueError(
                            "The impulse responses at receiver {} are {}: mix responses of one format, or turn mix_sources off.".format(r, " and ".join(task_formats))
                        )
                    auraliser = Auraliser([ir for ir, _ in task], anechoic_sounds=[sound for _, sound in task])
                    renders.append((group["branch"], auraliser, auraliser.convolve(), suffix, task_formats[0]))
        gains = Auraliser.render_gains([render[2] for render in renders], level, -1.0)
        signals = []
        peak = -np.inf
        for (branch, auraliser, signal, suffix, ir_format), gain in zip(renders, gains):
            leveled = signal * gain
            peak = max(peak, Auraliser.peak_dbfs(leveled))
            signals.append({"branch": branch, "audio": Audio(leveled, auraliser.sampling_rate, ir_format)})
            if save_folder:
                auraliser.export_convolved_sound(os.path.join(save_folder, f"{save_name or 'convolved_sound'}_mic_{suffix}.wav"), signal, gain=gain)
    return {"signals": signals, "peak_dbfs": float(peak), "warnings": messages}


def save_impulse_responses(groups, save_folder, level="relative", save_name=None):
    """``{"files": [str], "warnings": [str]}``: the impulse responses of ``groups``, as :func:`auralise` takes them,
    written to ``save_folder`` as ``<save_name or 'impulse_response'>_mic_<r>_<j>.wav`` with their JSON metadata,
    numbered as :func:`auralise` numbers its signals per source and receiver. ``level`` as in :func:`auralise`, on the
    raw responses."""
    level = _checked_level(level)
    named = [(f"{r}_{j}", ir) for group in groups for r, pairs in enumerate(_receivers(group)) for j, (ir, _) in enumerate(pairs)]
    gains = Auraliser.render_gains([ir.denormalized() for _, ir in named], level, -1.0)
    with _user_warnings() as messages:
        files = [str(ir.to_wav(os.path.join(save_folder, f"{save_name or 'impulse_response'}_mic_{suffix}.wav"), gain=gain)[0]) for (suffix, ir), gain in zip(named, gains)]
    return {"files": files, "warnings": messages}


_PLAYER = None


def _player():
    global _PLAYER
    if _PLAYER is None:
        _PLAYER = AudioPlayer()
    return _PLAYER


def play(sound, normalize=True, gain_db=0.0, start=0.0):
    """Play ``sound`` in the worker: an :class:`Audio` signal, or the path of an audio file.

    An Audio signal plays from ``start`` seconds with ``gain_db`` after the normalisation; a file plays whole.
    """
    if isinstance(sound, Audio):
        if sound.channels > 2:
            raise ValueError(
                "A {}-channel ambisonic signal cannot be played here. Save it with Auralise as a multichannel file for an "
                "ambisonic decoder, or simulate with Simulate Binaural for headphone playback.".format(sound.channels)
            )
        args = (sound.samples, sound.sampling_rate, normalize is not False, float(gain_db or 0.0), float(start or 0.0))
    elif isinstance(sound, str):
        if not os.path.exists(sound):
            raise ValueError(f"The file {sound} does not exist.")
        args = (sound,)
    else:
        raise TypeError(f"A sound is an Audio signal or a file path, not {type(sound).__name__}.")
    _player().play(*args)


def stop():
    """Stop what :func:`play` plays, and return to the beginning."""
    _player().stop()


def pause():
    """Pause the signal :func:`play` plays; :func:`resume` plays on."""
    _player().pause()


def resume():
    _player().resume()


def playback_status():
    """``{"playing", "position"}`` of what :func:`play` plays, the position in seconds."""
    player = _player()
    return {"playing": bool(player.playing), "position": float(player.position)}


def sound_samples(path):
    """An audio file as :class:`Audio`, mono, binaural (2 channels) or ambisonic, so that it shows as a signal does."""
    if not os.path.exists(path):
        raise ValueError(f"The file {path} does not exist.")
    try:
        samples, rate = sf.read(path, dtype="float64", always_2d=True)
    except Exception as e:
        raise ValueError(f"Cannot read {path}: {e}")
    channels = samples.shape[1]
    return Audio(samples[:, 0] if channels == 1 else samples, rate, "mono" if channels == 1 else "binaural" if channels == 2 else "ambisonic")


def sound_info(path):
    """The channels of an audio file, in any format soundfile reads: ``{"channels"}``."""
    if not os.path.exists(path):
        raise ValueError(f"The file {path} does not exist.")
    try:
        header = sf.info(path)
    except Exception as e:
        raise ValueError(f"Cannot read the channels of {path}: {e}")
    return {"channels": int(header.channels)}


# =============================================================================
# Rays and reflections: what Visualise Rays and Visualise Reflections trace; Regularized Fitness
# =============================================================================


def _index_message(kind, index, count):
    return "No {} at index {}; the scene has {} {}(s), 0 to {}.".format(kind, index, count, kind, count - 1)


def ray_field(scene, source_index=None, n_rays=500, max_order=10, direction=None, angle=None, scattering=False, seed=None, order=None):
    """``{"field": RayField, "shown": RayField, "seed": int, "warnings": [str]}``, as Visualise Rays traces them.

    ``shown`` is the field cut after ``order`` reflections, or the whole field. Only scattered rays are random; no
    ``seed`` draws one.
    """
    count = len(scene.sources)
    if source_index is not None and not 0 <= source_index < count:
        raise ValueError(_index_message("source", source_index, count))
    seed = random.randrange(2**31) if seed is None else int(seed)
    cone = {}
    if direction is not None:
        cone["direction"] = list(direction)
    if angle is not None:
        cone["angle"] = angle
    with _user_warnings() as messages:
        field = RayField.from_scene(
            scene,
            sources=None if source_index is None else [scene.sources[source_index]],
            n_rays=500 if n_rays is None else int(n_rays),
            max_reflection_order=int(max_order),
            include_scattering=bool(scattering),
            random_seed=seed,
            **cone,
        )
    shown = field if order is None else field.truncated(reflection_order=int(order))
    return {"field": field, "shown": shown, "seed": seed, "warnings": messages}


def energy_map(scene, points, source_index=None):
    """``{"spl": [[dB per band] per point], "spl_a": [dB(A) per point], "n_rays": int, "warnings": [str]}`` over
    ``points``, as Visualise Energy Map colours it; NaN outside the room.

    The settings are fixed so that the same scene gives the same map: image sources up to order 1, the rays of
    :func:`compas_acoustics.energy_map.map_rays`, seeded with 42.
    """
    with _user_warnings() as messages:
        result = EnergyMap.from_scene(scene, points, source_index=None if source_index is None else int(source_index), seed=True)
    return {
        "spl": result.spl().tolist(),
        "spl_a": result.spl_a().tolist(),
        "n_rays": result.n_rays,
        "warnings": messages,
    }


def specular_reflections(scene, orders=None, source_index=None, receiver_index=None):
    """``{"reflections": SpecularReflections, "counts": [[order, n]], "warnings": [str]}`` of the given orders, as
    Visualise Reflections finds them; negative orders are left out with a warning, no order is order 1."""
    notes = []
    orders = sorted({int(o) for o in orders or [] if o is not None})
    negative = [o for o in orders if o < 0]
    if negative:
        notes.append("Orders below 0 are left out: {}.".format(", ".join(str(o) for o in negative)))
        orders = [o for o in orders if o >= 0]
    orders = orders or [1]
    for kind, index, count in (("source", source_index, len(scene.sources)), ("receiver", receiver_index, len(scene.receivers))):
        if index is not None and not 0 <= index < count:
            notes.append(_index_message(kind, index, count))
    with _user_warnings() as messages:
        found = SpecularReflections.from_scene(scene, max_order=max(orders))
    reflections = found.filter(source=source_index, receiver_index=receiver_index, order=orders)
    counts = [[o, sum(1 for path in reflections if path.order == o)] for o in reflections.orders]
    return {"reflections": reflections, "counts": counts, "warnings": notes + messages}


def regularized_fitness(values, target, k=1.0):
    return float(_regularized_fitness(list(values), target, 1.0 if k is None else k))
