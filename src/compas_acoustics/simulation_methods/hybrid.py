"""The hybrid simulation method: image sources, optional edge diffraction, a directional FDN tail; ambisonic or binaural.

A :class:`~compas_acoustics.simulation_methods.SimulationMethod` on the ported hybrid-rir
modules of :mod:`compas_acoustics.hybrid`. The image source search runs once
for every source and receiver of the scene; each pair is then synthesised by
:func:`compas_acoustics.hybrid.synthesis.synthesize`.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from compas_acoustics.hybrid import UPSTREAM_REVISION
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.late import default_num_delays
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.hybrid.scattering import complementary_scattering
from compas_acoustics.hybrid.synthesis import synthesize
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.air import air_or_default
from compas_acoustics.room.geometry import pra_octave_grid
from compas_acoustics.room.pra import directivity_to_pra
from compas_acoustics.room.pra import source_to_pra

from .base import SimulationContext
from .base import SimulationMethod
from .base import _seed
from .base import _warn_if_the_search_is_slow
from .pra import Pyroomacoustics
from .raytracing import DURATION_RT_FACTOR
from .raytracing import default_duration

__all__ = ["Hybrid", "TAILS", "OUTPUTS"]

# A given reference must last this many times its reverberation time (the fitted or the Eyring one, whichever is longer)
MIN_REFERENCE_RT_FACTOR = 1.0

#: Where the decay target of the late field comes from.
TAILS = ("reference", "ray_tracing", "eyring", "sabine")

#: The formats the hybrid renders, and the IR graph attribute each is stored under.
OUTPUTS = {"ambisonic": "hybrid", "binaural": "hybrid_binaural"}

# The 8 kHz octave's upper edge must lie below Nyquist for pyFDN's band estimator.
_MIN_SAMPLING_RATE = 2.0 * 8000.0 * np.sqrt(2.0)


class Hybrid(SimulationMethod):
    """Image sources, optional edge diffraction and a directional FDN tail, rendered ambisonic and/or binaural.

    Image sources up to ``image_source_order`` (plus, with ``diffraction``, edge
    diffraction around obstacles and at the edges of the room's boundary that
    open wider than 180 deg) form the early field.
    The last of those orders also drives a feedback delay network with one
    delay line per direction, whose decay follows a target (Wendt et al. 2014;
    RAZR). Ported from hybrid-rir (gpapadimitri/hybrid-rir). Needs the
    ``hybrid`` extra and a closed room.

    One run renders either output, or both, from the same paths and delay
    lines (:class:`~compas_acoustics.hybrid.synthesis.HybridResult`). The
    ambisonic output encodes every path, and projects every line, at
    ``sh_order``. The binaural output filters every early path, and every delay
    line, with the HRIR measured nearest its direction
    (:func:`compas_acoustics.hybrid.binaural.binaural_rir`), as RAVEN renders its image
    sources and the directivity groups of its ray-tracing histogram, and RAZR
    its feedback delay network.

    Parameters
    ----------
    reference : str, dict, Pyroomacoustics or None
        A monaural response per pair, on the package scale, for the decay target
        (and, with ``late_envelope``, the late-envelope reference): the name of
        a method on the scene's IR graph (``"pyroomacoustics"``, ``"data"``), a
        dict of responses as :meth:`Scene.simulate` returns, or a
        :class:`~compas_acoustics.simulation_methods.Pyroomacoustics` method to run first.
    tail : {None, "reference", "ray_tracing", "eyring", "sabine"}
        Where the decay target comes from. ``"reference"``: fitted to
        ``reference``. ``"ray_tracing"``: fitted to a reference the hybrid runs
        itself, ``Pyroomacoustics(ray_tracing=True)`` with its default settings
        (image source order 3, 20000 rays) and the hybrid's ``air_absorption``. ``"eyring"``, ``"sabine"``:
        predicted from volume, area and mean absorption, without any reference;
        warns where the room is not diffuse. ``None`` means ``"reference"`` when
        a reference is given, else ``"ray_tracing"``.
    sh_order : int, default=1
        Ambisonic order of the ambisonic output. The network has 96 delay lines
        up to third order, the count Kirsch et al. (2023) use for offline simulation
        (:data:`~compas_acoustics.hybrid.late.MIN_DELAYS`), and the smallest
        equal-area grid supporting the order above that. The network input has
        the gain ``sqrt(num_delays)``, which offsets the ``1 / sqrt(num_delays)``
        of the output, so the late level does not depend on the count.
    image_source_order : int, default=3
        Highest image source order rendered as early field.
    normalization : {"SN3D", "N3D"}, default="SN3D"
        Output normalization, ACN channel order, W = 1 for a unit plane wave.
    n_slopes : int, default=1
        Decay slopes fitted to the reference; 0 lets DecayFitNet decide (needs
        ``multislope``). An Eyring or Sabine tail has one.
    decay_db : float, default=30.0
        Single-slope fit range from -5 dB: 30 is T30, 20 is T20.
    duration : float, optional
        Length in seconds; defaults to 1.5 times the longest target RT. A
        ray-traced reference the hybrid makes itself has the pyroomacoustics
        method's default length (1.5 times the longest Eyring RT), whatever the
        output's; the directional histogram takes ``duration``. A reference
        shorter than its reverberation time warns.
    directional_absorption : bool, default=True
        Weight each delay line with the reflection of the boundary it faces.
    late_envelope : bool, optional
        Shape the late field: attenuated per band to the reference's late
        energy, and each delay line given its direction's share of the energy
        the rays bring, RAVEN's directional histogram
        (:func:`compas_acoustics.simulation_methods.raytracing.directional_histograms`,
        :func:`compas_acoustics.hybrid.envelope.shape_late_directional`). The
        histogram is traced by the hybrid itself, ``Pyroomacoustics(ray_tracing=True)``
        with its default settings at ``image_source_order``, also when a
        reference is given. By default on for a reference or ray-tracing tail;
        an Eyring or Sabine tail has no reference, so there it is off and
        ``True`` raises.
    diffraction : {False, True, "udfa", "btms"}, default=False
        Edge diffraction around obstacles and at the edges of the room's boundary that
        open wider than 180 deg; ``True`` is ``"udfa"``. ``"btms"`` is
        slow, minutes per pair.
    air_absorption : bool, optional
        Air absorption on the image sources and diffracted paths. By default the
        reference's setting, or True without one. It must agree with the
        reference's, because the early field and the decay target share it.
    seed : bool or int, default=True
        Seeds the feedback matrices and delay jitter. True seeds with 42, so that the same scene gives the same responses;
        an int seeds with that value; False (or None) leaves them unseeded, a new draw each run, recorded as ``generation["seed"] = None``.
    normalize : bool, default=True
        Whether the returned responses are peak-normalized views.
    air : AirConditions, optional
        The air: speed of sound and ISO 9613-1 absorption, for the image
        sources, the diffracted paths, the predicted decay and the reference the
        hybrid runs itself. By default the references' air, else ``AirConditions()``
        (20 C, 50 %, 101325 Pa); a reference that records none counts as that.
        It must agree with the references', because the early field and the
        decay target share it.
    scattering : bool, default=False
        Surface scattering after Ewert et al. (2025), as RAZR 0.96.3 implements
        it: each image keeps its specular part, the product of ``sqrt(1 - s)``
        over its reflections, in the early field; its scattered part, the
        product of ``sqrt(s)``, enters the delay lines on its own hemisphere
        through an allpass cascade. Needs a second image source search.
    output : {"binaural", "ambisonic"} or list of them, default="binaural"
        The formats rendered, both from one run. :meth:`compute` returns the
        first, and :meth:`Scene.simulate` stores it under :attr:`attribute`;
        the second is in :attr:`extra_responses`. Binaural responses are stored
        under ``'hybrid_binaural'``, ambisonic ones under ``'hybrid'``; each
        records its format in ``generation['output']``.
    hrtf : str or path, optional
        The HRTF set of the binaural output: ``"kemar"``, the Auraliser's KEMAR
        set, or a SOFA file. ``None`` is ``"kemar"`` when the output is
        binaural; with an ambisonic output only it must stay ``None``.

    Attributes
    ----------
    reference_rirs : dict or None
        After :meth:`compute`, the references used: given, read from the graph, or run.
    results : dict
        After :meth:`compute`, the :class:`~compas_acoustics.hybrid.synthesis.HybridResult` per pair.
    outputs : tuple of str
        The formats rendered, in the order given.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> responses = scene.simulate(ca.Hybrid(tail="eyring"))  # doctest: +SKIP
    >>> ir = next(iter(responses.values()))  # doctest: +SKIP
    >>> ir.n_channels, ir.generation["output"], ir.generation["hrtf"]  # doctest: +SKIP
    (2, 'binaural', 'KEMAR_FreeFieldComp_48kHz.sofa')

    Both formats from one run: the ambisonic responses are returned and stored
    under ``'hybrid'``, the binaural ones under ``'hybrid_binaural'``.

    >>> responses = scene.simulate(ca.Hybrid(tail="eyring", output=["ambisonic", "binaural"]))  # doctest: +SKIP
    >>> next(iter(responses.values())).ambisonic_order, len(scene.impulse_responses("hybrid_binaural"))  # doctest: +SKIP
    (1, 1)

    With a ray-traced reference the decay is fitted to a response you already have.

    >>> reference = scene.simulate(ca.Pyroomacoustics(ray_tracing=True))  # doctest: +SKIP
    >>> responses = scene.simulate(ca.Hybrid(reference="pyroomacoustics"))  # doctest: +SKIP
    """

    name = "hybrid"

    def __init__(self, reference=None, tail=None, sh_order=1, image_source_order=3, normalization="SN3D",
                 n_slopes=1, decay_db=30.0, duration=None, directional_absorption=True, late_envelope=None, diffraction=False,
                 air_absorption=None, seed=True, normalize=True, air=None, scattering=False,
                 output="binaural", hrtf=None):
        if tail is not None and tail not in TAILS:
            raise ValueError(f"tail must be None or one of {TAILS}, not {tail!r}")
        if reference is not None and not isinstance(reference, (str, dict, Pyroomacoustics)):
            raise ValueError("reference must be a method name, a dict of impulse responses or a Pyroomacoustics method")
        if tail == "reference" and reference is None:
            raise ValueError("tail='reference' needs a reference")
        if tail in ("ray_tracing", "eyring", "sabine") and reference is not None:
            raise ValueError(f"tail={tail!r} takes no reference; pass tail='reference' or leave tail out to use it")
        if tail in ("eyring", "sabine"):
            if n_slopes != 1:
                raise ValueError(f"an {tail} tail has one decay slope; n_slopes must be 1, not {n_slopes!r}")
            if late_envelope:
                raise ValueError(f"late_envelope follows a reference; an {tail} tail has none")
        model = {False: None, True: "udfa", "udfa": "udfa", "btms": "btms"}.get(diffraction, "invalid")
        if model == "invalid":
            raise ValueError(f"diffraction must be False, True, 'udfa' or 'btms', not {diffraction!r}")
        if normalization not in ("SN3D", "N3D"):
            raise ValueError(f"normalization must be 'SN3D' or 'N3D', not {normalization!r}")
        if sh_order < 0:
            raise ValueError("sh_order must be non-negative")
        if image_source_order < 0:
            raise ValueError("image_source_order must be non-negative")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive")
        outputs = (output,) if isinstance(output, str) else tuple(output)
        if not outputs or len(set(outputs)) != len(outputs) or not set(outputs) <= set(OUTPUTS):
            raise ValueError(f"output must be one of {tuple(OUTPUTS)} or a list of them without repeats, not {output!r}")
        if "binaural" not in outputs and hrtf is not None:
            raise ValueError("hrtf is the HRTF set of the binaural output; add 'binaural' to output, or leave hrtf out")
        if "binaural" in outputs:
            hrtf = "kemar" if hrtf is None else hrtf
            if hrtf != "kemar" and not Path(hrtf).is_file():
                raise ValueError(f"hrtf must be 'kemar' or a SOFA file, not {hrtf!r}")

        self.reference = reference
        self.tail = tail if tail is not None else ("reference" if reference is not None else "ray_tracing")
        self.sh_order = int(sh_order)
        self.image_source_order = int(image_source_order)
        self.normalization = normalization
        self.n_slopes = n_slopes
        self.decay_db = decay_db
        self.duration = duration
        self.directional_absorption = directional_absorption
        self.late_envelope = self.tail not in ("eyring", "sabine") if late_envelope is None else bool(late_envelope)
        self.diffraction = model
        self.air_absorption = air_absorption
        self.seed = _seed(seed)
        self.normalize = normalize
        self.air = air
        self.scattering = bool(scattering)
        self.output = output
        self.outputs = outputs
        self.hrtf = hrtf
        self.reference_rirs = None
        self.results = {}
        self._extra_responses = {}

    @property
    def attribute(self) -> str:
        """``'hybrid'`` when the first output is ambisonic, ``'hybrid_binaural'`` when it is binaural."""
        return OUTPUTS[self.outputs[0]]

    @property
    def extra_responses(self) -> dict:
        """After :meth:`compute`, the second output's responses under its attribute, if there is a second."""
        return self._extra_responses

    # ------------------------------------------------------------------ run --
    def compute(self, context: SimulationContext) -> dict:
        self._extra_responses = {}
        fs = context.sampling_rate
        geometry = context.geometry
        if fs <= _MIN_SAMPLING_RATE:
            raise ValueError(
                f"the hybrid method needs a sample rate above 22627 Hz, got {fs}: the 8 kHz octave's "
                "upper edge must lie below Nyquist for the decay estimator"
            )
        if not geometry.is_closed:
            where = f"; first open edge from {geometry.open_edge[0]} to {geometry.open_edge[1]}" if geometry.open_edge else ""
            raise ValueError(f"the hybrid method needs boundary surfaces that close a volume{where}.")
        if context.outside:
            # As upstream RoomSpec: in a closed room a point outside is a modelling error.
            raise ValueError(f"the hybrid method: {context.outside} lie outside the room")
        # Before any work: a receiver facing along its up vector has no orientation
        rotations = {id(receiver): receiver.rotation for receiver in context.receivers}
        # A fully absorbing surface (an open window, alpha = 1) has no reflection, which the directional
        # absorption filters cannot represent when a delay line sees nothing else; 1 - 1e-6 reflects -60 dB.
        geometry = replace(geometry, materials={
            name: replace(material, absorption=np.minimum(material.absorption, 1.0 - 1e-6))
            for name, material in geometry.materials.items()
        })
        if not context.sources or not context.receivers:
            return {}

        _warn_if_the_search_is_slow(geometry, self.image_source_order)
        references, air_absorption, air, description = self._references(context)
        self.reference_rirs = references

        # With ray tracing on, pyroomacoustics' image damping holds prod sqrt(1 - s): the specular parts.
        # The same search in the room scattering 1 - s gives the scattered parts (complementary_scattering).
        rooms = [geometry.to_pra(fs, self.image_source_order, ray_tracing=self.scattering, air_absorption=False, air=air)]
        if self.scattering:
            rooms.append(complementary_scattering(geometry).to_pra(
                fs, self.image_source_order, ray_tracing=True, air_absorption=False, air=air))
        with pra_octave_grid():
            for pra_room in rooms:
                for source in context.sources:
                    pra_room.add_soundsource(source_to_pra(source))
                pra_room.add_microphone_array(
                    np.array([receiver.position for receiver in context.receivers], dtype=float).T
                )
                pra_room.image_source_model()
        room = rooms[0]
        scattered_room = rooms[1] if self.scattering else None
        speed_of_sound = float(getattr(room, "c", 343.0))
        # The network needs an int, ``rng + slope``: without a seed, a fresh one from the OS
        rng = self.seed if self.seed is not None else int(np.random.default_rng().integers(2**31))
        num_delays = default_num_delays(self.sh_order)
        generation = {
            "method": self.name,
            "scale": "package",
            "valid_range_hz": None,
            "tail": self.tail,
            "reference": description,
            "air_absorption": air_absorption,
            "air": air.to_dict(),
            "speed_of_sound": speed_of_sound,
            "image_source_order": self.image_source_order,
            "sh_order": self.sh_order,
            "n_slopes": self.n_slopes,
            "decay_db": self.decay_db,
            "diffraction": self.diffraction,
            "late_envelope": self.late_envelope,
            "directional_absorption": self.directional_absorption,
            "scattering": self.scattering,
            "num_delays": num_delays,
            "seed": self.seed,
            "scene": getattr(context.scene, "name", None),
            "upstream": f"hybrid-rir @ {UPSTREAM_REVISION}",
        }
        hrtf_path = None
        if "binaural" in self.outputs:
            from compas_acoustics.hybrid.binaural import DEFAULT_HRTF
            from compas_acoustics.hybrid.binaural import HrtfSet
            from compas_acoustics.hybrid.binaural import binaural_rir

            hrtf_path = DEFAULT_HRTF if self.hrtf == "kemar" else Path(self.hrtf)
            hrtf = HrtfSet.from_sofa(hrtf_path, fs)

        # The longest Eyring RT, against which a given reference's length is checked
        eyring_rt = default_duration(geometry, air_absorption, air=air, factor=1.0)

        histograms = None
        if self.late_envelope:
            from compas_acoustics.hybrid.late import fdn_directions
            from compas_acoustics.simulation_methods.raytracing import directional_histograms

            grid = fdn_directions(self.sh_order, num_delays).cartesian
            grid = grid / np.linalg.norm(grid, axis=1, keepdims=True)
            tracer = Pyroomacoustics(ray_tracing=True)
            histograms = directional_histograms(
                geometry,
                [source_to_pra(source) for source in context.sources],
                np.array([receiver.position for receiver in context.receivers], dtype=float).T,
                fs,
                [grid @ rotations[id(receiver)].T for receiver in context.receivers],
                max_order=self.image_source_order,
                duration=self._own_duration(geometry, air_absorption, air, output_length=True),
                n_rays=tracer.n_rays,
                receiver_radius=tracer.receiver_radius,
                energy_threshold=tracer.energy_thres,
                air_absorption=air_absorption,
                seed=self.seed,
                air=air,
            )

        rirs, self.results = {output: {} for output in self.outputs}, {}
        for s, source in enumerate(context.sources):
            directivity = (
                None if source.directivity == "omnidirectional"
                else directivity_to_pra(source.directivity, source.direction)
            )
            for m, receiver in enumerate(context.receivers):
                key = context.key(source, receiver)
                spec = Pair(
                    geometry=geometry,
                    source=np.asarray(source.position, dtype=float),
                    receiver=np.asarray(receiver.position, dtype=float),
                    receiver_rotation=rotations[id(receiver)],
                    fs=float(fs),
                    speed_of_sound=speed_of_sound,
                    air_absorption=air_absorption,
                    air=air,
                )
                reference = None
                if references is not None:
                    # Decay target and late envelope need the absolute (package) level
                    reference = np.asarray(references[key].denormalized(), dtype=float)
                    target = RoomTarget.from_rir(reference, fs, n_slopes=self.n_slopes, decay_db=self.decay_db)
                    self._check_reference_length(reference.size / fs, target, eyring_rt, key)
                else:
                    target = RoomTarget.from_room(spec, model=self.tail)
                result = synthesize(
                    spec,
                    image_sources(room, s, m, spec, directivity, scattered_room),
                    target,
                    max_order=self.image_source_order,
                    sh_order=self.sh_order,
                    duration_s=self.duration,
                    num_delays=num_delays,
                    directional_absorption=self.directional_absorption,
                    late_envelope=reference if self.late_envelope else None,
                    directional_envelope=histograms[m][s] if histograms is not None else None,
                    diffraction=self.diffraction or False,
                    normalization=self.normalization,
                    rng=rng,
                    scattering=self.scattering,
                    hrtf=hrtf if "binaural" in rirs else None,
                )
                self.results[key] = result
                if "ambisonic" in rirs:
                    rirs["ambisonic"][key] = ImpulseResponse(
                        name=key, source=source, receiver=receiver, ir=result.rir, sampling_rate=fs,
                        normalized=self.normalize,
                        generation=dict(generation, output="ambisonic"), ambisonic_order=self.sh_order,
                        normalization=self.normalization.lower(), channel_order="acn",
                    )
                if "binaural" in rirs:
                    rirs["binaural"][key] = ImpulseResponse(
                        name=key, source=source, receiver=receiver, ir=binaural_rir(result, hrtf), sampling_rate=fs,
                        normalized=self.normalize, generation=dict(generation, output="binaural", hrtf=hrtf_path.name),
                    )
        self._extra_responses = {OUTPUTS[output]: rirs[output] for output in self.outputs[1:]}
        return rirs[self.outputs[0]]

    # ----------------------------------------------------------- references --
    def _own_duration(self, geometry, air_absorption, air, output_length: bool = False):
        """Length of the runs the hybrid makes itself: the pyroomacoustics method's default (``DURATION_RT_FACTOR``
        times the longest Eyring RT). A ray-traced reference always carries its full decay, whatever ``duration`` the
        output has; the directional histogram (``output_length``) needs no more than the output and takes ``duration``."""
        if output_length and self.duration is not None:
            return self.duration
        return default_duration(geometry, air_absorption, air=air)

    def _check_reference_length(self, length, target, eyring_rt, key):
        """Warn when a reference ends before its own reverberation time: pyFDN's fit then shrinks its range to what
        the response has, and the late envelope reads a cut tail. Measured: a reference of 1.0 times the Eyring RT
        read T30 at 125 Hz 25 % short in a room with curtain walls; 1.5 times gave the 2.5-times result."""
        needed = MIN_REFERENCE_RT_FACTOR * max(float(np.max(target.rt)), float(eyring_rt))
        if length < needed:
            warnings.warn(
                f"The reference of {key} is {length:.1f} s long, shorter than its reverberation time ({needed:.1f} s): the "
                f"decay fit and the late envelope read a cut tail. Simulate the reference with time_thres (duration) of at "
                f"least {DURATION_RT_FACTOR * needed / MIN_REFERENCE_RT_FACTOR:.1f} s.",
                UserWarning,
                stacklevel=3,
            )

    def _references(self, context):
        """``(references or None, air_absorption, air, description)`` for :attr:`tail`."""
        if self.tail in ("eyring", "sabine"):
            absorption = True if self.air_absorption is None else bool(self.air_absorption)
            return None, absorption, air_or_default(self.air), None

        if self.tail == "ray_tracing":
            absorption = True if self.air_absorption is None else bool(self.air_absorption)
            method = Pyroomacoustics(ray_tracing=True, air_absorption=absorption, air=self.air,
                                     duration=self._own_duration(context.geometry, absorption, self.air))
            references, description = method.compute(context), method.generation
        elif isinstance(self.reference, str):
            references, description = context.scene.impulse_responses(self.reference), self.reference
        elif isinstance(self.reference, Pyroomacoustics):
            references, description = self.reference.compute(context), self.reference.generation
        else:
            references, description = self.reference, "dict"

        fs = context.sampling_rate
        for source in context.sources:
            for receiver in context.receivers:
                key = context.key(source, receiver)
                if key not in references:
                    hint = (f"; run a method named {self.reference!r} on this scene first" if isinstance(self.reference, str)
                            else "; references are matched by source and receiver, so they must come from this scene")
                    raise ValueError(f"the reference has no response for '{receiver.name}' and '{source.name}'{hint}")
                response = references[key]
                if response.sampling_rate != fs:
                    raise ValueError(f"reference '{key}' has sample rate {response.sampling_rate}; the simulation runs at {fs}")
                if response.n_channels != 1:
                    raise ValueError(f"reference '{key}' is not a mono response; pass a Pyroomacoustics result")
                if response.generation.get("scale") != "package":
                    raise ValueError(f"reference '{key}' is not on the package scale; pass a Pyroomacoustics result")
        pairs = [references[context.key(s, r)] for s in context.sources for r in context.receivers]
        if any(r.generation.get("method") == "pyroomacoustics" and not r.generation.get("ray_tracing") for r in pairs):
            warnings.warn(
                "the reference is image sources only, without ray tracing: it has no diffuse decay to fit. "
                "Use Pyroomacoustics(ray_tracing=True), or tail='eyring'.",
                UserWarning,
                stacklevel=3,
            )

        used = {r.generation.get("air_absorption") for r in pairs}
        if self.air_absorption is None:
            if len(used) != 1 or None in used:
                raise ValueError(f"the references disagree about air_absorption ({used}); pass air_absorption")
            absorption = used.pop()
        else:
            absorption = bool(self.air_absorption)
            if used != {absorption}:
                raise ValueError(
                    f"the references were simulated with air_absorption={used}, not {absorption}; "
                    "the early field and the decay target must agree"
                )
        conditions = {air_or_default(AirConditions.from_dict(r.generation.get("air"))) for r in pairs}
        if self.air is None:
            if len(conditions) != 1:
                raise ValueError(f"the references disagree about the air ({conditions}); pass air")
            air = conditions.pop()
        else:
            air = self.air
            if conditions != {air}:
                raise ValueError(
                    f"the references were simulated in the air {conditions}, not {air}; "
                    "the early field and the decay target must share it"
                )
        return references, absorption, air, description
