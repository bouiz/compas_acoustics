"""Functions the worker serves, called through :func:`compas_acoustics.worker.client` or the Grasshopper plugin.

Arguments and results are what :mod:`compas_acoustics.pb` carries: Data objects, dicts, lists and numbers. Unusable
input raises ValueError or TypeError, which the service answers with its message.
"""

import functools
import importlib.util
import math
import os
import random
import re
import threading
import warnings
from collections import OrderedDict
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
    "make_scattering_sample",
    "run_scattering_lab",
    "polar_balloon",
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


def _template(text):
    """A pattern that matches the messages the format template ``text`` makes: each replacement field matches any
    text."""
    return re.compile(re.sub(r"\\\{[^{}]*?\\\}", ".+?", re.escape(text)) + r"\Z", re.DOTALL)


def _split_notes(messages, notes, dropped=()):
    """``messages`` as ``(warnings, notes)``: those that one of the format templates ``notes`` makes are the notes;
    those of ``dropped`` go."""

    def made_by(message, templates):
        return any(_template(t).match(message) for t in templates)

    kept = [m for m in messages if not made_by(m, dropped)]
    return [m for m in kept if not made_by(m, notes)], [m for m in kept if made_by(m, notes)]


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


def make_source(name, position, direction=None, directivity=None, spl=None):
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


def analysis(irs, time_scale="db"):
    """``{"analyses": [analysis], "warnings": [str]}``, one analysis per response of ``irs`` in order, as Analyse IR
    shows them.

    An analysis is ``{"parameters": [parameter], "time": parameter, "frequency": parameter}``; a parameter is
    ``{"name", "title", "x_unit", "y_unit", "x_scale" ('bands', 'linear' or 'log'), "frequencies", "values" (one list
    per channel), "overall", "channels"}``; STI, G, SPL and LF are left out where all their values are NaN, LFC always.
    ``time_scale`` ``'db'`` gives the time response in dB re its peak, ``'linear'`` its amplitude to 60 dB down its
    decay.
    """
    time_scale = time_scale or TIME_SCALES[0]
    if time_scale not in TIME_SCALES:
        raise ValueError(f"Input parameter 'time_scale' must be one of {TIME_SCALES}.")
    analyses = []
    with _user_warnings() as messages:
        for ir in irs:
            analyser = Analyser(ir)
            analyser.analyse()
            data = analyser.analysis_data
            bands = analyser.frequency_bands
            parameters = []
            for name, (title, x_unit, y_unit) in PARAMETERS.items():
                if name == "LFC":
                    continue
                values = [float(v) for v in data[name]]
                if name in ("STI", "G", "SPL", "LF") and all(math.isnan(v) for v in values):
                    continue
                parameters.append(_parameter(name, title, x_unit, y_unit, "bands", bands, [values], data[f"{name}_overall"]))
            samples, channels = _plotted_channels(ir)
            db = time_scale == "db"
            times, levels = time_response(samples, ir.sampling_rate, db=db, trim_db=None if db else 60.0)
            time = _parameter("IR", _curve_title("Impulse Response", channels), "Time (s)", "Level (dB)" if db else "Amplitude", "linear", times, levels, channels=channels)
            frequencies, magnitudes = frequency_response(samples, ir.sampling_rate, num_fractions=3, frequency_range=(20.0, 20000.0))
            title = _curve_title("Frequency Response, 1/3 octave", channels)
            frequency = _parameter("FR", title, "Frequency (Hz)", "Level (dB)", "log", frequencies, magnitudes, channels=channels)
            analyses.append({"parameters": parameters, "time": time, "frequency": frequency})
    return {"analyses": analyses, "warnings": messages}


# =============================================================================
# Auralisation and playback: what Auralise and the Audio Player do
# =============================================================================

LEVELS = ["relative", "individual", "none"]


def _checked_level(level):
    level = level or LEVELS[0]
    if level not in LEVELS:
        raise ValueError(f"Input parameter 'level' must be one of {LEVELS}.")
    return level


# the bundled anechoic recordings auralise takes by name, the files of the package's anechoic sound library
_ANECHOIC_SOUNDS = ("female_speech", "male_speech", "viola", "guitar", "castanets")


def auralise(irs, sound, chosen, labels, level="relative", save_folder=None, save_name=None):
    """``{"signals": [{"label", "audio": Audio}], "peak_dbfs": float, "warnings": [str]}``: the chosen impulse
    responses convolved with the anechoic ``sound``, one signal per chosen response in the order of ``chosen``.

    ``sound`` is the name of a bundled recording (``'female_speech'``, ``'male_speech'``, ``'viola'``, ``'guitar'`` or
    ``'castanets'``) or the path of a sound file.

    ``irs`` holds every connected response, ``chosen`` the indices of those to render and ``labels`` one label per
    chosen response. ``level``: ``'relative'`` (one gain from the sound convolved with every response of ``irs``, the
    loudest at -1 dBFS, so a chosen response renders as in a run of all of them), ``'individual'`` or ``'none'``
    (only the chosen responses are convolved). With a ``save_folder`` each signal is also written there as
    ``<save_name or 'convolved_sound'>_mic_<label>.wav``.
    """
    level = _checked_level(level)
    if sound in _ANECHOIC_SOUNDS:
        sound = os.path.join(Defaults.ANECHOIC_SOUNDFILES_LIBRARY_PATH, sound + ".wav")
    if not os.path.exists(sound):
        raise ValueError(f"The sound file {sound} does not exist. Bundled sounds: {', '.join(_ANECHOIC_SOUNDS)}.")
    irs = list(irs)
    relative = level == "relative"
    with _user_warnings() as messages:
        kept = set(chosen)
        auralisers = {}
        signals = {}
        peaks = []
        for k, index in enumerate(range(len(irs)) if relative else chosen):
            jobs.progress("rendering", k)
            auraliser = Auraliser([irs[index]], anechoic_sounds=[sound])
            signal = auraliser.convolve()
            if index in kept:
                auralisers[index], signals[index] = auraliser, signal
            # the shared gain needs only each pair's peak: an unchosen pair's signal is not kept
            if relative:
                peaks.append(np.max(np.abs(signal), initial=0.0))
        if relative:
            shared = Auraliser.render_gains(peaks, level, -1.0)[0]
            gains = {index: shared for index in chosen}
        else:
            gains = dict(zip(chosen, Auraliser.render_gains([signals[index] for index in chosen], level, -1.0)))
        rendered = []
        peak = -np.inf
        for index, label in zip(chosen, labels):
            leveled = signals[index] * gains[index]
            peak = max(peak, Auraliser.peak_dbfs(leveled))
            rendered.append({"label": label, "audio": Audio(leveled, auralisers[index].sampling_rate, irs[index].format)})
            if save_folder:
                auralisers[index].export_convolved_sound(os.path.join(save_folder, f"{save_name or 'convolved_sound'}_mic_{label}.wav"), signals[index], gain=gains[index])
    return {"signals": rendered, "peak_dbfs": float(peak), "warnings": messages}


def save_impulse_responses(irs, chosen, labels, save_folder, level="relative", save_name=None):
    """``{"files": [str], "warnings": [str]}``: the chosen impulse responses of ``irs``, as :func:`auralise` takes
    them, written to ``save_folder`` as ``<save_name or 'impulse_response'>_mic_<label>.wav`` with their JSON
    metadata. ``level`` as in :func:`auralise`, on the raw responses: ``'relative'`` takes one gain from every
    response of ``irs``."""
    level = _checked_level(level)
    irs = list(irs)
    raw = [ir.denormalized() for ir in irs] if level == "relative" else [irs[index].denormalized() for index in chosen]
    gains = Auraliser.render_gains(raw, level, -1.0)
    if level == "relative":
        gains = [gains[index] for index in chosen]
    with _user_warnings() as messages:
        files = [
            str(irs[index].to_wav(os.path.join(save_folder, f"{save_name or 'impulse_response'}_mic_{label}.wav"), gain=gain)[0])
            for index, label, gain in zip(chosen, labels, gains)
        ]
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


# =============================================================================
# The coefficient lab: what Scattering Sample, Scattering Lab and Polar Preview do
# =============================================================================

SAMPLE_MODES = ["as_drawn", "disc"]
BAND_RANGES = ["iso", "extended"]
BALLOON_BODIES = ["sample", "reference"]


def make_scattering_sample(mesh, mode="as_drawn", diameter=None, reference=None, baseplate=0.01, symmetry="none", period=None, name=None, frame=None):
    """A :class:`ScatteringSample` of a relief, as the Scattering Sample component makes it.

    The lab measures a relief that lies on the world XY plane, its mounting surface at z = 0 and the sound coming from
    +z; give ``frame`` for an object that lies elsewhere.

    Parameters
    ----------
    mesh : :class:`compas.datastructures.Mesh`
        The relief: the tile to repeat in ``'disc'`` mode, the whole relief as drawn otherwise.
    mode : str
        ``'as_drawn'`` or ``'disc'``.
    diameter : float, optional
        The disc's diameter in m; ``'disc'`` mode only, and required there.
    reference : :class:`compas.datastructures.Mesh`, optional
        A proxy mesh to take as the reference instead of the flat footprint; ``'as_drawn'`` mode only.
    baseplate : float
        The plate thickness in m below the relief.
    symmetry : str
        ``'none'`` or ``'rotational'``.
    period : sequence of float, optional
        The tile period (x, y) in m of the ``'disc'`` mode; the tile's bounding box when None or empty.
    name : str, optional
    frame : :class:`compas.geometry.Frame`, optional
        The mounting plane of the object: its origin on the mounting surface and its Z axis pointing out into the room,
        where the sound comes from. ``mesh`` and ``reference`` are moved from it to the world XY plane before anything
        else. None when the object lies on the world XY plane facing +z.

    Returns
    -------
    :class:`compas_acoustics.lab.ScatteringSample`

    Raises
    ------
    ValueError
        For an unknown mode, a disc without a diameter, a ``period`` that is not 2 numbers, a ``frame`` that is not a
        frame, or an object whose footprint on the XY plane has no area.
    """
    from compas.geometry import Frame
    from compas.geometry import Transformation

    from compas_acoustics.lab import ScatteringSample

    mode = mode or SAMPLE_MODES[0]
    if mode not in SAMPLE_MODES:
        raise ValueError(f"mode is 'as_drawn' or 'disc', not {mode!r}")
    symmetry = symmetry or "none"
    baseplate = 0.01 if baseplate is None else baseplate
    if frame is not None:
        if not isinstance(frame, Frame):
            raise ValueError(f"frame is the mounting plane as a Frame, not {type(frame).__name__}")
        to_world = Transformation.from_frame_to_frame(frame, Frame.worldXY())
        mesh = mesh.transformed(to_world)
        reference = reference.transformed(to_world) if reference is not None else None
    if mode == "disc":
        if diameter is None:
            raise ValueError("disc mode needs a 'diameter'")
        if not diameter > 0:
            raise ValueError(f"'diameter' must be greater than 0, not {diameter!r}")
        period = list(period) if period is not None else []
        if period and len(period) != 2:
            raise ValueError(f"period is 2 numbers, the tile's x and y, not {len(period)}")
        sample = ScatteringSample.disc(mesh, diameter, period=tuple(period) if period else None, baseplate=baseplate, symmetry=symmetry)
    else:
        sample = ScatteringSample.as_drawn(mesh, baseplate=baseplate, symmetry=symmetry, reference=reference)
    if name:
        sample.name = name
    return sample


def run_scattering_lab(sample, method="geometric", preset=None, band_range="iso", settings=None):
    """Measure the scattering of ``sample`` with :func:`compas_acoustics.lab.measure_scattering`, as the Scattering Lab
    component does.

    Parameters
    ----------
    sample : :class:`compas_acoustics.lab.ScatteringSample`
    method : str
        ``'geometric'``, ``'kirchhoff'`` or ``'bem'``.
    preset : str, optional
        ``'survey'`` or ``'standard'``; the method's own when None or empty.
    band_range : str
        ``'iso'`` (100 to 5000 Hz) or ``'extended'`` (to 8000 Hz).
    settings : dict, optional
        The method's options, passed on as keyword arguments (the geometric method's ``n_rays``, for one).

    Returns
    -------
    dict
        ``{"result": ScatteringResult, "scattering_coefficients": [7 floats], "octave_bands": [int], "warnings": [str],
        "notes": [str]}``: the coefficients on the material's 7 octaves 125 to 8000 Hz, and the octaves that were
        simulated. ``notes`` holds what states what the run is and flags no fault (the geometric method's heuristic, the
        octaves not simulated, bands beyond ISO 17497, Kirchhoff falling back to one process); ``warnings`` holds the
        rest. Neither holds the sample's own warnings (the Scattering Sample shows them) nor how long the run takes,
        which is news only before it.

    Raises
    ------
    ValueError
        For an unknown ``band_range``, or a lab that cannot run here (the geometric method without trimesh and embreex:
        the message names ``method="kirchhoff"``, which needs neither).
    """
    from compas_acoustics.lab import EXTENDED_BAND_RANGE
    from compas_acoustics.lab import ISO_BAND_RANGE
    from compas_acoustics.lab import measure_scattering
    from compas_acoustics.lab.grids import _SLOW_GRID
    from compas_acoustics.lab.scattering import DURATION_TEMPLATES
    from compas_acoustics.lab.scattering import NOTE_TEMPLATES
    from compas_acoustics.lab.solvers.bem import _BIG
    from compas_acoustics.lab.solvers.kirchhoff import _SERIAL

    band_range = band_range or BAND_RANGES[0]
    ranges = {"iso": ISO_BAND_RANGE, "extended": EXTENDED_BAND_RANGE}
    if band_range not in ranges:
        raise ValueError(f"band_range is 'iso' or 'extended', not {band_range!r}")
    with _user_warnings() as caught:
        try:
            result = measure_scattering(sample, method=method or "geometric", preset=preset or None, band_range=ranges[band_range], **dict(settings or {}))
        except ImportError as error:
            # the service answers a ValueError with its message and an ImportError with a traceback
            raise ValueError(str(error)) from error
    # the sample's own warnings are the Scattering Sample's; how long the run takes is news only before it
    messages = [m for m in caught if m not in sample.warnings]
    messages, notes = _split_notes(messages, NOTE_TEMPLATES + (_SERIAL,), dropped=DURATION_TEMPLATES + (_SLOW_GRID, _BIG))
    _hold(result)
    return {"result": result, "scattering_coefficients": result.scattering_coefficients, "octave_bands": list(result.octave_bands), "warnings": messages, "notes": notes}


#: The start of the error of a call that names a result by its guid when the worker does not hold it (it restarted, or
#: let it go to stay within _HELD_BYTES): the caller sends the result itself instead
NOT_HELD = "result not held:"
_HELD_BYTES = 1024 * 2**20  # the polar levels the held results may take in all; the newest result is held whatever its size
_held = OrderedDict()  # guid -> ScatteringResult, the least recently used first
_held_lock = threading.Lock()  # the service answers calls on threads


def _polar_bytes(result):
    return int(result.polar_sample_db.nbytes + result.polar_reference_db.nbytes)


def _hold(result):
    """Keep ``result`` by its guid, so that Polar Preview can name it instead of sending it again; let the least
    recently used go."""
    with _held_lock:
        _held[str(result.guid)] = result
        _held.move_to_end(str(result.guid))
        while len(_held) > 1 and sum(_polar_bytes(r) for r in _held.values()) > _HELD_BYTES:
            _held.popitem(last=False)


def _the_result(result):
    """``result`` itself, or the held result its guid names; ``ValueError`` starting with :data:`NOT_HELD` when that is
    not held."""
    if not isinstance(result, str):
        _hold(result)
        return result
    with _held_lock:
        if result not in _held:
            raise ValueError(f"{NOT_HELD} {result}")
        _held.move_to_end(result)
        return _held[result]


_BALLOON_MIN_RADIUS = 0.03  # the fraction of the radius at and below the floor: a balloon that collapses to a point is not a mesh


def polar_balloon(result, preview=0, octave=1000, body="sample", radius=1.0, floor_db=-30.0):
    """The polar response of one preview source and octave as a balloon mesh, as the Polar Balloon component shows it.

    Each receiver direction is moved out to ``radius * (r_min + (1 - r_min) * clip(1 - level / floor_db, 0, 1))`` with
    ``r_min = 0.03``: the level 0 dB, the maximum of the flat reference, at ``radius``, a level of ``floor_db`` or below
    (the empty cells of a sparse geometric result are at -300 dB) at ``r_min * radius``, and linear in between. The
    radius never reaches 0, so that no two vertices coincide and the mesh stays valid.

    Parameters
    ----------
    result : :class:`compas_acoustics.lab.ScatteringResult` or str
        The result, or the guid of one the worker holds: one :func:`run_scattering_lab` made or a call was given. A guid
        the worker does not hold is a ``ValueError`` starting with :data:`NOT_HELD`, and the caller sends the result
        itself.
    preview : int
        The index into ``result.polar_source_indices``.
    octave : int
        An octave of ``result.octave_bands``.
    body : str
        ``'sample'`` or ``'reference'``.
    radius : float
        The radius of the loudest direction (0 dB).
    floor_db : float
        A negative level in dB: a direction at this level or below is at ``0.03 * radius``.

    Returns
    -------
    dict
        ``{"mesh": Mesh, "levels": [dB per vertex], "source": [x, y, z], "label": str}``: ``source`` is the unit vector
        towards the preview source times ``1.2 * radius``.

    Raises
    ------
    ValueError
        For a ``preview``, ``octave`` or ``body`` the result does not have (the message lists the valid values), or a
        ``floor_db`` that is not negative.
    """
    import numpy as np
    from compas.datastructures import Mesh
    from scipy.spatial import ConvexHull

    result = _the_result(result)
    previews = result.polar_source_indices
    if preview is None or not 0 <= int(preview) < len(previews):
        raise ValueError(f"preview is an index of the preview sources, 0 to {len(previews) - 1}, not {preview!r}")
    preview = int(preview)
    if octave is None or int(octave) not in result.octave_bands:
        raise ValueError(f"octave is one of the simulated octaves {list(result.octave_bands)} Hz, not {octave!r}")
    octave = int(octave)
    body = body or BALLOON_BODIES[0]
    if body not in BALLOON_BODIES:
        raise ValueError(f"body is 'sample' or 'reference', not {body!r}")
    floor_db = -30.0 if floor_db is None else float(floor_db)
    if not floor_db < 0:
        raise ValueError(f"floor_db is a negative level in dB, not {floor_db!r}")
    radius = 1.0 if radius is None else float(radius)
    if not radius > 0:
        raise ValueError(f"radius is a positive length, not {radius!r}")

    polar = result.polar_sample_db if body == "sample" else result.polar_reference_db
    levels = np.asarray(polar[preview, result.octave_bands.index(octave)], dtype=float)
    directions = np.asarray(result.receiver_directions, dtype=float)
    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    fraction = np.clip(1.0 - levels / floor_db, 0.0, 1.0)
    vertices = directions * (radius * (_BALLOON_MIN_RADIUS + (1.0 - _BALLOON_MIN_RADIUS) * fraction))[:, np.newaxis]

    # The directions cover a hemisphere: a point below it closes the hull, and the faces that use it are dropped
    n = len(directions)
    hull = ConvexHull(np.vstack([directions, [0.0, 0.0, -1.0]]))
    faces = []
    for simplex, outward in zip(hull.simplices, hull.equations[:, :3]):
        if n in simplex:
            continue
        a, b, c = directions[simplex]
        faces.append([int(i) for i in simplex] if np.cross(b - a, c - a) @ outward > 0 else [int(simplex[0]), int(simplex[2]), int(simplex[1])])
    mesh = Mesh.from_vertices_and_faces(vertices.tolist(), faces)

    index = previews[preview]
    colatitude, azimuth = float(result.source_colatitude[index]), float(result.source_azimuth[index])
    theta, phi = np.radians(colatitude), np.radians(azimuth)
    source = 1.2 * radius * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return {"mesh": mesh, "levels": [float(v) for v in levels], "source": [float(v) for v in source], "label": f"θ {colatitude:g}°, φ {azimuth:g}°, {octave} Hz, {body}"}
