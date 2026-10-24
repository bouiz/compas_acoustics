"""Functions the worker serves; called through :func:`compas_acoustics.worker.client` or the Grasshopper plugin, never directly.

Arguments and results are what compas_pb carries (:mod:`compas_acoustics.pb`): compas_acoustics and COMPAS Data
objects, dicts, lists and numbers. A function raises ValueError or TypeError for input it cannot use; the service
answers those with their message.
"""

import importlib.util
import sys
import warnings

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
    "analysis",
    "auralise",
    "play",
    "stop",
    "ray_field",
    "energy_map",
    "specular_reflections",
    "regularized_fitness",
    "sound_info",
    "pause",
    "resume",
    "playback_status",
    "sound_samples",
    "make_scattering_sample",
    "run_scattering_lab",
    "polar_balloon",
    "lab_material",
]


def _method_class(name):
    if name in METHODS:
        return METHODS[name]
    if name in ("binaural", "ambisonics", "mono"):
        from compas_acoustics import simulation_methods

        return getattr(simulation_methods, name.capitalize())
    raise ValueError(f"Unknown method {name!r}: 'binaural', 'ambisonics' or 'mono'.")


#: Further simulation methods by name, besides ``binaural``, ``ambisonics`` and ``mono``
METHODS = {}


def _messages(caught):
    return list(dict.fromkeys(str(w.message) for w in caught if issubclass(w.category, UserWarning)))


def info():
    import compas_pb

    from compas_acoustics import Defaults
    from compas_acoustics import __version__
    from compas_acoustics.pb import SCHEMA
    from compas_acoustics.worker import fingerprint

    return {
        "python": sys.version.split()[0],
        "compas_acoustics": __version__,
        "fingerprint": fingerprint(),
        "schema": SCHEMA,
        "compas_pb": compas_pb.__version__,
        "speed_of_sound": Defaults().DEFAULT_SPEED_OF_SOUND,
        "hybrid": importlib.util.find_spec("pyFDN") is not None,
    }


def simulate(scene, method, settings, references=None):
    """Simulate ``scene`` with ``method`` and ``settings``.

    Parameters
    ----------
    scene : :class:`compas_acoustics.Scene`
    method : str
        ``'binaural'``, ``'ambisonics'`` or ``'mono'``.
    settings : dict
        The method's keyword arguments.
    references : dict, optional
        ``{key: ImpulseResponse}``, the method's ``reference``, keyed as :meth:`SimulationContext.key` keys a pair.

    Returns
    -------
    dict
        ``{"responses": {name: {key: ImpulseResponse}}, "warnings": [str]}``: the responses under the method's name, the
        graph attribute they are stored under.
    """
    method_class = _method_class(method)
    settings = dict(settings or {})
    if references:
        settings["reference"] = dict(references)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        method = method_class(**settings)
        rirs = scene.simulate(method)
    return {"responses": {method.name: dict(rirs)}, "warnings": _messages(caught)}


def analyse(ir):
    """STI and G of ``ir``, for a client whose pyrato cannot compute them: ``{"analysis": {...}, "warnings": [str]}``."""
    from compas_acoustics import AcousticAnalyser

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # the client computes every other parameter itself
        analysis = AcousticAnalyser(ir)._sti_and_strength()
    return {"analysis": analysis, "warnings": _messages(caught)}


# =============================================================================
# Materials and the model: what the Material, Source, Receiver, Acoustic Object and Acoustic Scene components make
# =============================================================================


def materials_library():
    """The materials library's names by category: ``[[category, [name, ...]], ...]``, in the library's category order."""
    from compas_acoustics import MaterialsLibrary

    return [[category.value, list(names)] for category, names in MaterialsLibrary().materials_by_category.items()]


def material_from_name(name):
    from compas_acoustics import MaterialsLibrary

    library = MaterialsLibrary()
    if name not in library.material_names:
        raise ValueError(f"No material {name!r} in the materials library.")
    return library.material_from_name(name)


def make_material(name, absorption, scattering=None, category=None):
    """An :class:`AcousticMaterial`; an empty ``scattering`` or ``category`` is left to the class's default."""
    from compas_acoustics import AcousticMaterial

    return AcousticMaterial(name=name, absorption_coefficients=list(absorption), scattering_coefficients=list(scattering) if scattering else None,
                            category=category or None)


#: The source directivities, as Source and Directivity offer them
DIRECTIVITIES = ["bidirectional", "hypercardioid", "supercardioid", "cardioid", "omnidirectional"]


def make_source(name, position, direction=None, directivity=None, sound_file=None, spl=None):
    """A :class:`Source` with the defaults of the Source component: ``my_source``, facing +x, omnidirectional.

    ``spl``, the source's level on its axis at 1 m per band (one value for every band), keeps the default when None.
    """
    from compas_acoustics import Source

    directivity = directivity or "omnidirectional"
    if directivity not in DIRECTIVITIES:
        raise ValueError("Input parameter 'directivity' is not valid.")
    source = Source(name or "my_source", *(position if position is not None else [0, 0, 0]), directivity=directivity,
                    direction=list(direction) if direction is not None else [1, 0, 0], sound_file=sound_file or None)
    if spl is not None and len(list(spl)) > 0:
        source.spl = spl
    return source


def make_receiver(name, position=None, direction=None, frame=None):
    """An upright :class:`Receiver` (+z up), as the Receiver component makes it.

    A ``frame`` sets the position and orientation together (x axis facing, z axis up); ``frame_overrides`` says
    whether ``position`` or ``direction`` were given too and are not used.

    Returns
    -------
    dict
        ``{"receiver": Receiver, "frame_overrides": bool}``
    """
    from compas.geometry import Vector

    from compas_acoustics import Receiver

    name = name or "my_receiver"
    if frame is not None:
        receiver = Receiver.from_frame(name, frame)
    else:
        receiver = Receiver(name, *(position if position is not None else [0, 0, 1]))
        if direction is not None:
            receiver.direction = Vector(*direction)
    # raises for a receiver facing straight up or down, whose orientation is undefined
    receiver.frame
    return {"receiver": receiver, "frame_overrides": frame is not None and any(v is not None for v in (position, direction))}


def make_acoustic_object(geometry, material, name=None, role="boundary"):
    """An :class:`AcousticObject` of a Mesh or a Polyline."""
    from compas.datastructures import Mesh
    from compas.geometry import Polyline

    from compas_acoustics import AcousticObject

    if not isinstance(geometry, (Mesh, Polyline)):
        raise TypeError("Unsupported geometry type.")
    return AcousticObject(acoustic_geometry=geometry, material=material, name=name or None, role=role or "boundary")


def make_scene(objects, sources, receivers, name=None, sampling_rate=None):
    """A :class:`Scene` of the objects, sources and receivers; an item of the wrong type is a TypeError that names it."""
    from compas_acoustics import AcousticObject
    from compas_acoustics import Receiver
    from compas_acoustics import Scene
    from compas_acoustics import Source

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


# =============================================================================
# Analysis: what Analyse IR shows
# =============================================================================

#: name: (title, x unit, y unit), the parameters Analyse IR gives. The bands of STI are its modulation transfer indices.
PARAMETERS = {
    "EDT": ("Early Decay Time (EDT)", "Frequency (Hz)", "Time (s)"),
    "T20": ("Reverberation Time (T20)", "Frequency (Hz)", "Time (s)"),
    "T30": ("Reverberation Time (T30)", "Frequency (Hz)", "Time (s)"),
    "C50": ("Clarity (C50)", "Frequency (Hz)", "dB"),
    "C80": ("Clarity (C80)", "Frequency (Hz)", "dB"),
    "D50": ("Definition (D50)", "Frequency (Hz)", "Value (%)"),
    "DRR": ("Direct-to-Reverberant Ratio (DRR)", "Frequency (Hz)", "dB"),
    "Ts": ("Center Time (Ts)", "Frequency (Hz)", "Time (s)"),
    "STI": ("Speech Transmission Index (STI)", "Frequency (Hz)", "Index (0-1)"),
    "G": ("Sound Strength (G)", "Frequency (Hz)", "dB"),
    "SPL": ("Sound Pressure Level (SPL), overall A-weighted", "Frequency (Hz)", "dB"),
    "LF": ("Lateral Energy Fraction (LF)", "Frequency (Hz)", "Fraction (0-1)"),
    "LFC": ("Lateral Energy Fraction, cosine (LFC)", "Frequency (Hz)", "Fraction (0-1)"),
}
TIME_SCALES = ["db", "linear"]
# The frequency axis of a frequency response
FREQUENCY_AXIS = (20.0, 20000.0)
# An amplitude plot of a time response ends where its Schroeder curve falls this far, in dB
LINEAR_TIME_TRIM_DB = 60.0


def _parameter(name, title, x_unit, y_unit, x_scale, x, values, overall=None, channels=()):
    return {"name": name, "title": title, "x_unit": x_unit, "y_unit": y_unit, "x_scale": x_scale,
            "frequencies": [float(v) for v in x], "values": [[float(v) for v in row] for row in values],
            "overall": None if overall is None else float(overall), "channels": list(channels)}


def _plotted_channels(ir):
    """The raw channels a curve shows, (samples, channels), and their names: W of an ambisonic response, both ears of a binaural one."""
    import numpy as np

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
    """The room acoustic parameters of ``ir`` per band, its time response and its frequency response, as Analyse IR shows them.

    Parameters
    ----------
    ir : :class:`compas_acoustics.ImpulseResponse`
    time_scale : str
        ``'db'``: the time response as its level in dB re the peak; ``'linear'``: its amplitude, ending 60 dB down its decay.

    Returns
    -------
    dict
        ``{"parameters": [parameter], "time": parameter, "frequency": parameter, "warnings": [str]}``. A parameter is
        ``{"name", "title", "x_unit", "y_unit", "x_scale" ('bands', 'linear' or 'log'), "frequencies", "values" (one list
        per channel), "overall", "channels"}``. STI and G are left out where they could not be computed (all NaN).
    """
    import math

    from compas_acoustics import AcousticAnalyser
    from compas_acoustics.analyser import frequency_response
    from compas_acoustics.analyser import time_response

    time_scale = time_scale or TIME_SCALES[0]
    if time_scale not in TIME_SCALES:
        raise ValueError(f"Input parameter 'time_scale' must be one of {TIME_SCALES}: 'db' or 'linear'.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyser = AcousticAnalyser(ir)
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
        times, levels = time_response(samples, ir.sampling_rate, db=db, trim_db=None if db else LINEAR_TIME_TRIM_DB)
        time = _parameter("IR", _curve_title("Impulse Response", channels), "Time (s)", "Level (dB)" if db else "Amplitude", "linear",
                          times, levels, channels=channels)
        frequencies, magnitudes = frequency_response(samples, ir.sampling_rate, num_fractions=3, frequency_range=FREQUENCY_AXIS)
        frequency = _parameter("FR", _curve_title("Frequency Response, 1/3 octave", channels), "Frequency (Hz)", "Level (dB)", "log",
                               frequencies, magnitudes, channels=channels)
    return {"parameters": parameters, "time": time, "frequency": frequency, "warnings": _messages(caught)}


# =============================================================================
# Auralisation and playback: what Auralise and the Audio Player do
# =============================================================================

LEVELS = ["relative", "individual", "none"]
HEADROOM_DB = -1.0


def auralise(groups, level="relative", mix_sources=False, save_folder=None, save_name=None):
    """Convolve impulse responses with anechoic sounds, per receiver, as Auralise does.

    Parameters
    ----------
    groups : list of dict
        One per Grasshopper branch: ``{"branch": str, "irs": [ImpulseResponse], "sounds": [path or None]}``; a None
        sound is the sound file of the response's source.
    level : str
        ``'relative'`` (one gain for every signal of the call, the loudest at -1 dBFS), ``'individual'`` or ``'none'``.
    mix_sources : bool
        One signal per receiver, its sources summed; else one per source and receiver.
    save_folder, save_name : str, optional
        With a folder, each signal is also written there as ``<save_name or 'convolved_sound'>_mic_<suffix>.wav``.

    Returns
    -------
    dict
        ``{"signals": [{"branch", "audio": Audio, "suffix"}], "peak_dbfs": float, "warnings": [str]}``.
    """
    import os

    import numpy as np

    from compas_acoustics.auraliser import Auraliser
    from compas_acoustics.pb.types import Audio
    from compas_acoustics.worker import jobs

    level = level or LEVELS[0]
    if level not in LEVELS:
        raise ValueError(f"Input parameter 'level' must be one of {LEVELS}.")
    renders = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for group in groups:
            irs = list(group["irs"])
            sounds = list(group.get("sounds") or [None] * len(irs))
            if len(sounds) != len(irs):
                raise ValueError("DataTree branch length mismatch.")
            ir_groups, sound_groups, formats = {}, {}, {}
            for idx, (ir, sound) in enumerate(zip(irs, sounds)):
                receiver_id = str(ir.receiver.guid)
                sub_key = f"source_{idx}"
                formats[sub_key] = ir.format
                # Auralise at the raw level so level differences between impulse responses are kept
                raw = ir.denormalized()
                data = raw[:, np.newaxis] if raw.ndim == 1 else raw
                order = getattr(ir, "ambisonic_order", None)
                ir_groups.setdefault(receiver_id, {})[sub_key] = {
                    "ir": data,
                    "sampling_rate": ir.sampling_rate,
                    "ambisonic_order": int(round(np.sqrt(data.shape[1]))) - 1 if order is None else int(order),
                    "channel_order": getattr(ir, "channel_order", None) or "acn",
                    "normalization": getattr(ir, "normalization", None) or "sn3d",
                    "binaural": ir.format == "binaural",
                }
                sound_groups.setdefault(receiver_id, {})[sub_key] = {"filepath": sound or ir.source.sound_file}
            for r, receiver_id in enumerate(ir_groups):
                if mix_sources:
                    tasks = [(receiver_id, ir_groups[receiver_id], sound_groups[receiver_id], f"{r}")]
                else:
                    tasks = [(key, {key: ir_groups[receiver_id][key]}, {key: sound_groups[receiver_id][key]}, f"{r}_{j}")
                             for j, key in enumerate(ir_groups[receiver_id])]
                for name, ir_data, sound_data, suffix in tasks:
                    jobs.progress("rendering", len(renders))
                    task_formats = sorted({formats[key] for key in ir_data})
                    if len(task_formats) > 1:
                        raise ValueError("The impulse responses at receiver {} are {}: mix responses of one format, or turn mix_sources off.".format(
                            r, " and ".join(task_formats)))
                    auraliser = Auraliser(name=name, impulse_responses=ir_data, anechoic_sounds=sound_data)
                    renders.append((group["branch"], auraliser, auraliser.convolve(), suffix, task_formats[0]))
        gains = Auraliser.render_gains([render[2] for render in renders], level, HEADROOM_DB)
        signals = []
        peak = -np.inf
        for (branch, auraliser, signal, suffix, ir_format), gain in zip(renders, gains):
            leveled = signal * gain
            peak = max(peak, Auraliser.peak_dbfs(leveled))
            signals.append({"branch": branch, "audio": Audio(leveled, auraliser.sampling_rate, ir_format), "suffix": suffix})
            if save_folder:
                auraliser.export_convolved_sound(os.path.join(save_folder, f"{save_name or 'convolved_sound'}_mic_{suffix}.wav"), signal, gain=gain)
    return {"signals": signals, "peak_dbfs": float(peak), "warnings": _messages(caught)}


_PLAYER = None


def _player():
    global _PLAYER
    if _PLAYER is None:
        from compas_acoustics import AudioPlayer

        _PLAYER = AudioPlayer()
    return _PLAYER


def play(sound, normalize=True, gain_db=0.0, start=0.0):
    """Play ``sound`` in the worker: an :class:`Audio` signal, or the path of an audio file.

    An Audio signal plays from ``start`` seconds with ``gain_db`` after the normalisation; a file plays whole.

    Returns
    -------
    dict
        ``{"warnings": [str], "duration": seconds}``; 0 s for a file.
    """
    import os

    from compas_acoustics.pb.types import Audio

    if isinstance(sound, Audio):
        if sound.channels > 2:
            raise ValueError("A {}-channel ambisonic signal cannot be played here. Save it with Auralise as a multichannel file for an "
                             "ambisonic decoder, or simulate with Simulate Binaural for headphone playback.".format(sound.channels))
        args = (sound.samples, sound.sampling_rate, normalize is not False, float(gain_db or 0.0), float(start or 0.0))
    elif isinstance(sound, str):
        if not os.path.exists(sound):
            raise ValueError(f"The file {sound} does not exist.")
        args = (sound,)
    else:
        raise TypeError(f"A sound is an Audio signal or a file path, not {type(sound).__name__}.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        player = _player()
        player.play(*args)
    return {"warnings": _messages(caught), "duration": float(player.duration)}


def stop():
    """Stop what :func:`play` plays, and return to the beginning."""
    _player().stop()
    return {"warnings": []}


def pause():
    """Pause the signal :func:`play` plays; :func:`resume` plays on."""
    _player().pause()
    return {"warnings": []}


def resume():
    _player().resume()
    return {"warnings": []}


def playback_status():
    """``{"playing", "paused", "position", "duration"}`` of what :func:`play` plays, in seconds."""
    player = _player()
    return {"playing": bool(player.playing), "paused": bool(player.paused), "position": float(player.position),
            "duration": float(player.duration)}


def sound_samples(path):
    """The samples of an audio file as :class:`Audio`, so that a file shows as a signal does: mono, binaural (2 channels) or ambisonic."""
    import os

    import soundfile as sf

    from compas_acoustics.pb.types import Audio

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
    """Trace a ray field from the scene's sources, as Visualise Rays does.

    Returns
    -------
    dict
        ``{"field": RayField, "shown": RayField, "seed": int, "warnings": [str]}``: ``shown`` is the field cut after
        ``order`` reflections, or the whole field. Only scattered rays are random; no ``seed`` draws one.
    """
    import random

    from compas_acoustics import RayField

    count = len(scene.sources)
    if source_index is not None and not 0 <= source_index < count:
        raise ValueError(_index_message("source", source_index, count))
    seed = random.randrange(2**31) if seed is None else int(seed)
    cone = {}
    if direction is not None:
        cone["direction"] = list(direction)
    if angle is not None:
        cone["angle"] = angle
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        field = RayField.from_scene(
            scene,
            sources=None if source_index is None else [scene.sources[source_index]],
            n_rays=500 if n_rays is None else int(n_rays),
            max_reflection_order=10 if max_order is None else int(max_order),
            include_scattering=bool(scattering),
            random_seed=seed,
            **cone,
        )
    shown = field if order is None else field.truncated(reflection_order=int(order))
    return {"field": field, "shown": shown, "seed": seed, "warnings": _messages(caught)}


def energy_map(scene, points, source_index=None):
    """The sound pressure level of the scene's sources over ``points``, as Visualise Energy Map colours it.

    The settings are fixed, as Pachyderm's mapping fixes its own: the image sources up to order 1, the rays of
    :func:`compas_acoustics.energy_map.map_rays`, seeded with 42, so that the same scene gives the same map.

    Returns
    -------
    dict
        ``{"bands": [Hz], "inside": [bool], "spl": [[dB per band] per point], "spl_a": [dB(A) per point], "n_rays": int,
        "warnings": [str]}``: NaN for a point outside the room (:class:`compas_acoustics.EnergyMap`).
    """
    from compas_acoustics import EnergyMap

    count = len(scene.sources)
    if source_index is not None and not 0 <= source_index < count:
        raise ValueError(_index_message("source", source_index, count))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = EnergyMap.from_scene(scene, points, source_index=None if source_index is None else int(source_index), seed=True)
    return {"bands": result.bands.tolist(), "inside": [bool(v) for v in result.inside], "spl": result.spl().tolist(),
            "spl_a": result.spl_a().tolist(), "n_rays": result.n_rays, "warnings": _messages(caught)}


def specular_reflections(scene, orders=None, source_index=None, receiver_index=None):
    """The specular reflection paths of the given orders, as Visualise Reflections finds them.

    Returns
    -------
    dict
        ``{"reflections": SpecularReflections, "orders": [int], "counts": [[order, n]], "warnings": [str]}``.
        Negative orders are left out with a warning; no order is order 1.
    """
    from compas_acoustics import SpecularReflections

    messages = []
    orders = sorted({int(o) for o in orders or [] if o is not None})
    negative = [o for o in orders if o < 0]
    if negative:
        messages.append("Orders below 0 are left out: {}.".format(", ".join(str(o) for o in negative)))
        orders = [o for o in orders if o >= 0]
    orders = orders or [1]
    for kind, index, count in (("source", source_index, len(scene.sources)), ("receiver", receiver_index, len(scene.receivers))):
        if index is not None and not 0 <= index < count:
            messages.append(_index_message(kind, index, count))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = SpecularReflections.from_scene(scene, max_order=max(orders))
    reflections = found.filter(source=source_index, receiver_index=receiver_index, order=orders)
    counts = [[o, sum(1 for path in reflections if path.order == o)] for o in reflections.orders]
    return {"reflections": reflections, "orders": orders, "counts": counts, "warnings": messages + _messages(caught)}


def regularized_fitness(values, target, k=1.0):
    from compas_acoustics import regularized_fitness as fitness

    return float(fitness(list(values), target, 1.0 if k is None else k))


def sound_info(path):
    """The channels, sample rate and length of an audio file, in any format soundfile reads: ``{"channels", "sampling_rate", "frames"}``."""
    import os

    import soundfile as sf

    if not os.path.exists(path):
        raise ValueError(f"The file {path} does not exist.")
    try:
        info = sf.info(path)
    except Exception as e:
        raise ValueError(f"Cannot read the channels of {path}: {e}")
    return {"channels": int(info.channels), "sampling_rate": int(info.samplerate), "frames": int(info.frames)}


# =============================================================================
# The coefficient lab: what Scattering Sample, Scattering Lab, Polar Balloon and Lab Material do
# =============================================================================

SAMPLE_MODES = ["as_drawn", "disc"]
BAND_RANGES = ["iso", "extended"]
BALLOON_BODIES = ["sample", "reference"]


def make_scattering_sample(mesh, mode="as_drawn", diameter=None, reference=None, baseplate=0.01, symmetry="none", period=None, name=None, frame=None):
    """A :class:`ScatteringSample` of a relief, as the Scattering Sample component makes it.

    The lab measures a relief that lies on the world XY plane, its mounting surface at z = 0 and the sound coming from +z; give ``frame``
    for an object that lies elsewhere.

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
        The mounting plane of the object: its origin on the mounting surface and its Z axis pointing out into the room, where the sound
        comes from. ``mesh`` and ``reference`` are moved from it to the world XY plane before anything else. None when the object lies
        on the world XY plane facing +z.

    Returns
    -------
    :class:`compas_acoustics.lab.ScatteringSample`

    Raises
    ------
    ValueError
        For an unknown mode, a disc without a diameter, a ``period`` that is not 2 numbers, a ``frame`` that is not a frame, or an object
        whose footprint on the XY plane has no area.
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
            raise ValueError("a disc sample needs a diameter")
        if not diameter > 0:
            raise ValueError(f"diameter is a positive length in m, not {diameter!r}")
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
    """Measure the scattering of ``sample`` with :func:`compas_acoustics.lab.measure_scattering`, as the Scattering Lab component does.

    Parameters
    ----------
    sample : :class:`compas_acoustics.lab.ScatteringSample`
    method : str
        ``'geometric'``, ``'kirchhoff'`` or ``'bem'``.
    preset : str, optional
        ``'survey'``, ``'standard'`` or ``'fine'``; the method's own when None or empty.
    band_range : str
        ``'iso'`` (100 to 5000 Hz) or ``'extended'`` (to 8000 Hz).
    settings : dict, optional
        The method's options, passed on as keyword arguments (the geometric method's ``n_rays``, for one).

    Returns
    -------
    dict
        ``{"result": ScatteringResult, "scattering_coefficients": [7 floats], "octave_bands": [int], "warnings": [str]}``: the
        coefficients on the material's 7 octaves 125 to 8000 Hz, and the octaves that were simulated.

    Raises
    ------
    ValueError
        For an unknown ``band_range``, or a lab that cannot run here (the geometric method without trimesh and embreex: the message
        names ``method="kirchhoff"``, which needs neither).
    """
    from compas_acoustics.lab import EXTENDED_BAND_RANGE
    from compas_acoustics.lab import ISO_BAND_RANGE
    from compas_acoustics.lab import measure_scattering

    band_range = band_range or BAND_RANGES[0]
    ranges = {"iso": ISO_BAND_RANGE, "extended": EXTENDED_BAND_RANGE}
    if band_range not in ranges:
        raise ValueError(f"band_range is 'iso' or 'extended', not {band_range!r}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = measure_scattering(sample, method=method or "geometric", preset=preset or None, band_range=ranges[band_range], **dict(settings or {}))
        except ImportError as error:
            # the service answers a ValueError with its message and an ImportError with a traceback
            raise ValueError(str(error)) from error
    return {"result": result, "scattering_coefficients": result.scattering_coefficients, "octave_bands": list(result.octave_bands),
            "warnings": _messages(caught)}


def polar_balloon(result, preview=0, octave=1000, body="sample", radius=1.0, floor_db=-30.0):
    """The polar response of one preview source and octave as a balloon mesh, as the Polar Balloon component shows it.

    Each receiver direction is moved out to ``radius * max(0, 1 - level / floor_db)``: the level 0 dB, the maximum of the flat
    reference, at ``radius`` and a level of ``floor_db`` or below at the centre.

    Parameters
    ----------
    result : :class:`compas_acoustics.lab.ScatteringResult`
    preview : int
        The index into ``result.polar_source_indices``.
    octave : int
        An octave of ``result.octave_bands``.
    body : str
        ``'sample'`` or ``'reference'``.
    radius : float
    floor_db : float
        A negative level in dB.

    Returns
    -------
    dict
        ``{"mesh": Mesh, "levels": [dB per vertex], "source": [x, y, z], "label": str}``: ``source`` is the unit vector towards the preview
        source times ``1.2 * radius``.

    Raises
    ------
    ValueError
        For a ``preview``, ``octave`` or ``body`` the result does not have (the message lists the valid values), or a ``floor_db`` that
        is not negative.
    """
    import numpy as np
    from compas.datastructures import Mesh
    from scipy.spatial import ConvexHull

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
    vertices = directions * (radius * np.maximum(0.0, 1.0 - levels / floor_db))[:, np.newaxis]

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
    return {"mesh": mesh, "levels": [float(v) for v in levels], "source": [float(v) for v in source],
            "label": f"θ {colatitude:g}°, φ {azimuth:g}°, {octave} Hz, {body}"}


def lab_material(name, absorption, result=None, category=None):
    """An :class:`AcousticMaterial` of a given absorption and a measured scattering, as the Lab Material component makes it.

    Parameters
    ----------
    name : str
    absorption : sequence of float or :class:`AcousticMaterial`
        1 value for every octave, or 7 for 125 to 8000 Hz; or a material to take its 7 absorption coefficients from.
    result : :class:`compas_acoustics.lab.ScatteringResult`, optional
        The scattering to take the material's scattering coefficients from.
    category : str, optional

    Returns
    -------
    dict
        ``{"material": AcousticMaterial, "warnings": [str]}``: the warnings are those of :func:`compas_acoustics.lab.lab_material`, a
        coefficient clipped to [0, 1] for one.

    Raises
    ------
    ValueError
        If ``absorption`` is missing or empty, or not 1 or 7 values.
    """
    import numpy as np

    from compas_acoustics.lab import lab_material as make_lab_material

    if absorption is None or (isinstance(absorption, (list, tuple, np.ndarray)) and len(absorption) == 0):
        raise ValueError("absorption is required: 1 or 7 values, or a material to take them from")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        material = make_lab_material(name, absorption, scattering=result, category=category or None)
    messages = _messages(caught)
    if result is not None and result.generation.get("heuristic"):
        messages.append("s comes from the heuristic geometric method: use kirchhoff or bem for a measurement")
    return {"material": material, "warnings": messages}
