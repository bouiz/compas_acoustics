"""The hybrid methods :class:`Binaural` and :class:`Ambisonics`: image sources, optional edge diffraction, a directional FDN.

One image source search serves every pair; each pair is prepared by :func:`compas_acoustics.hybrid.synthesis.prepare`
and rendered by :func:`~compas_acoustics.hybrid.binaural.render_binaural` or
:func:`~compas_acoustics.hybrid.ambisonics.render_ambisonics`.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import replace
from pathlib import Path

import numpy as np

from compas_acoustics.hybrid import UPSTREAM_REVISION
from compas_acoustics.hybrid.ambisonics import default_num_delays
from compas_acoustics.hybrid.ambisonics import render_ambisonics
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.late import MIN_DELAYS
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.hybrid.scattering import complementary_scattering
from compas_acoustics.hybrid.synthesis import prepare
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.air import air_or_default
from compas_acoustics.room.geometry import pra_octave_grid
from compas_acoustics.room.pra import directivity_to_pra
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.raytracing import DURATION_RT_FACTOR
from compas_acoustics.room.raytracing import default_duration

from .base import SimulationContext
from .base import SimulationMethod
from .base import _seed
from .base import _warn_if_the_search_is_slow
from .pra import Mono

__all__ = ["Binaural", "Ambisonics", "TAILS"]

#: Where the decay target of the late field comes from.
TAILS = ("reference", "ray_tracing", "eyring", "sabine")


class _HybridMethod(SimulationMethod):
    """The references, image sources, network and directional histograms; a subclass sets ``num_delays`` and renders (:meth:`_response`)."""

    num_delays: int

    def __init__(self, reference=None, tail=None, image_source_order=3, n_slopes=1, decay_db=30.0, duration=None,
                 directional_absorption=True, late_envelope=None, diffraction=False, air_absorption=None, seed=True,
                 normalize=True, air=None, scattering=False):
        if tail is not None and tail not in TAILS:
            raise ValueError(f"tail must be None or one of {TAILS}, not {tail!r}")
        if reference is not None and not isinstance(reference, (str, dict, Mono)):
            raise ValueError("reference must be a method name, a dict of impulse responses or a Mono method")
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
        if image_source_order < 0:
            raise ValueError("image_source_order must be non-negative")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive")

        self.reference = reference
        self.tail = tail if tail is not None else ("reference" if reference is not None else "ray_tracing")
        self.image_source_order = int(image_source_order)
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
        self.reference_rirs = None
        self.results = {}

    @abstractmethod
    def _response(self, core, key, source, receiver, generation) -> ImpulseResponse:
        """Render ``core`` in the method's format, keep the rendering in :attr:`results`, and wrap it."""

    def compute(self, context: SimulationContext) -> dict:
        fs = context.sampling_rate
        geometry = context.geometry
        # the 8 kHz octave's upper edge must lie below Nyquist
        if fs <= 2.0 * 8000.0 * np.sqrt(2.0):
            raise ValueError(
                f"the hybrid method needs a sample rate above 22627 Hz, got {fs}: the 8 kHz octave's "
                "upper edge must lie below Nyquist for the decay estimator"
            )
        if not geometry.is_closed:
            where = f"; first open edge from {geometry.open_edge[0]} to {geometry.open_edge[1]}" if geometry.open_edge else ""
            raise ValueError(f"the hybrid method needs boundary surfaces that close a volume{where}.")
        if context.outside:
            # in a closed room a point outside is a modelling error
            raise ValueError(f"the hybrid method: {context.outside} lie outside the room")
        # Before any work: a receiver facing along its up vector has no orientation
        rotations = {id(receiver): receiver.rotation for receiver in context.receivers}
        # A fully absorbing surface (an open window, alpha = 1) has no reflection, which the directional
        # absorption filters cannot represent when a delay line sees nothing else; 1 - 1e-6 reflects -60 dB.
        geometry = replace(geometry, materials={
            name: replace(material, absorption=np.minimum(material.absorption, 1.0 - 1e-6))
            for name, material in geometry.materials.items()
        })
        self.results = {}
        if not context.sources or not context.receivers:
            return {}

        _warn_if_the_search_is_slow(geometry, self.image_source_order)
        from compas_acoustics.worker import jobs

        jobs.progress("references")
        references, air_absorption, air, description = self._references(context)
        self.reference_rirs = references

        # Built for ray tracing, an image's damping holds prod sqrt(1 - s), its specular part; the same search
        # in the room scattering 1 - s gives the scattered parts (complementary_scattering).
        jobs.progress("image sources")
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
            "n_slopes": self.n_slopes,
            "decay_db": self.decay_db,
            "diffraction": self.diffraction,
            "late_envelope": self.late_envelope,
            "directional_absorption": self.directional_absorption,
            "scattering": self.scattering,
            "num_delays": self.num_delays,
            "seed": self.seed,
            "scene": getattr(context.scene, "name", None),
            "upstream": f"hybrid-rir @ {UPSTREAM_REVISION}",
        }

        # The longest Eyring RT, against which a given reference's length is checked
        eyring_rt = default_duration(geometry, air_absorption, air=air, factor=1.0)

        histograms = None
        if self.late_envelope:
            from compas_acoustics.room.raytracing import directional_histograms

            jobs.progress("directional histograms")
            grid = fdn_directions(self.num_delays).cartesian
            grid = grid / np.linalg.norm(grid, axis=1, keepdims=True)
            tracer = Mono(ray_tracing=True)
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

        rirs = {}
        for s, source in enumerate(context.sources):
            directivity = (
                None if source.directivity == "omnidirectional"
                else directivity_to_pra(source.directivity, source.direction)
            )
            for m, receiver in enumerate(context.receivers):
                jobs.progress("rendering", s * len(context.receivers) + m, len(context.sources) * len(context.receivers))
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
                core = prepare(
                    spec,
                    image_sources(room, s, m, spec, directivity, scattered_room),
                    target,
                    max_order=self.image_source_order,
                    duration_s=self.duration,
                    num_delays=self.num_delays,
                    directional_absorption=self.directional_absorption,
                    late_envelope=reference if self.late_envelope else None,
                    directional_envelope=histograms[m][s] if histograms is not None else None,
                    diffraction=self.diffraction or False,
                    rng=rng,
                    scattering=self.scattering,
                )
                rirs[key] = self._response(core, key, source, receiver, generation)
        return rirs

    def _own_duration(self, geometry, air_absorption, air, output_length: bool = False):
        """Length of the runs the method makes itself: the mono method's default, ``DURATION_RT_FACTOR`` times the longest
        Eyring RT, so that a reference carries its full decay; the directional histogram (``output_length``) takes ``duration``."""
        if output_length and self.duration is not None:
            return self.duration
        return default_duration(geometry, air_absorption, air=air)

    def _check_reference_length(self, length, target, eyring_rt, key):
        """Warn when a reference ends before 2/3 of its reverberation time (the fitted or Eyring one, the longer).

        The decay fit then shrinks to what the response has and the late envelope reads a cut tail: 1.0 times the
        Eyring RT read T30 at 125 Hz 25 % short in a room with curtain walls; 1.5 times gave the 2.5-times result.
        """
        rt = max(float(np.max(target.rt)), float(eyring_rt))
        if length < 2.0 / 3.0 * rt:
            warnings.warn(
                f"The reference of {key} is {length:.1f} s long, less than 2/3 of its reverberation time ({rt:.1f} s): the "
                f"decay fit and the late envelope read a cut tail. Simulate the reference with a duration of at "
                f"least {DURATION_RT_FACTOR * rt:.1f} s.",
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
            method = Mono(ray_tracing=True, air_absorption=absorption, air=self.air,
                          duration=self._own_duration(context.geometry, absorption, self.air))
            references, description = method.compute(context), method.generation
        elif isinstance(self.reference, str):
            references, description = context.scene.impulse_responses(self.reference), self.reference
        elif isinstance(self.reference, Mono):
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
                    raise ValueError(f"reference '{key}' is not a mono response; pass a Mono result")
                if response.generation.get("scale") != "package":
                    raise ValueError(f"reference '{key}' is not on the package scale; pass a Mono result")
        pairs = [references[context.key(s, r)] for s in context.sources for r in context.receivers]
        if any(r.generation.get("method") == "mono" and not r.generation.get("ray_tracing") for r in pairs):
            warnings.warn(
                "the reference is image sources only, without ray tracing: it has no diffuse decay to fit. "
                "Use Mono(ray_tracing=True), or tail='eyring'.",
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


_ENGINE_PARAMETERS = """
    tail : {None, "reference", "ray_tracing", "eyring", "sabine"}
        Where the decay target comes from: fitted to ``reference``; fitted to a ``Mono(ray_tracing=True)`` run
        of its default settings with the method's ``air_absorption``; or predicted from volume, area and mean
        absorption (warns where the room is not diffuse). None: ``"reference"`` with a reference, else
        ``"ray_tracing"``.
    image_source_order : int, default=3
        The last order of the early field; it also drives the network.
    n_slopes : int, default=1
        Decay slopes fitted to the reference; 0 lets DecayFitNet (Goetz et al. 2022) decide (needs
        ``multislope``). An Eyring or Sabine tail has one.
    decay_db : float, default=30.0
        Single-slope fit range from -5 dB: 30 is T30, 20 is T20.
    duration : float, optional
        In seconds; default 1.5 times the longest target RT. A reference the method runs itself has the mono
        method's default length whatever the output's; the directional histogram takes ``duration``. A reference
        shorter than 2/3 of its reverberation time warns.
    directional_absorption : bool, default=True
        Weight each delay line with the reflection of the boundary it faces.
    late_envelope : bool, optional
        Shape the late field: attenuated per band to the reference's late energy, each delay line given its
        direction's share of the energy the rays bring (directional histogram, Schroeder 2011). The rays are
        traced by ``Mono(ray_tracing=True)`` at ``image_source_order``, also with a given reference. Default on
        for a reference or ray-tracing tail; an Eyring or Sabine tail has no reference, so ``True`` raises.
    diffraction : {False, True, "udfa", "btms"}, default=False
        Edge diffraction around obstacles and at room edges that open wider than 180 deg; ``True`` is
        ``"udfa"``. ``"btms"`` takes minutes per pair.
    air_absorption : bool, optional
        On the image sources and diffracted paths. Default the reference's setting, or True; it must agree with
        the reference's, since the early field and the decay target share it.
    seed : bool or int, default=True
        For the feedback matrices and delay jitter: True seeds with 42, an int with that value, False or None
        leaves them unseeded (``generation["seed"] = None``). One seed gives both methods the same late field.
    normalize : bool, default=True
        Whether the responses are peak-normalized views.
    air : AirConditions, optional
        Speed of sound and ISO 9613-1 absorption for the image sources, diffracted paths, predicted decay and the
        reference run. Default the references' air, else ``AirConditions()``; it must agree with the references'.
    scattering : bool, default=False
        Surface scattering after Ewert et al. (2025): each image keeps its specular part, ``prod sqrt(1 - s)``,
        in the early field; its scattered part, ``prod sqrt(s)``, enters the delay lines of its hemisphere
        through an allpass cascade. Needs a second image source search.

    Attributes
    ----------
    reference_rirs : dict or None
        After :meth:`compute`, the references used: given, read from the graph, or run.
    results : dict
        After :meth:`compute`, the :class:`~compas_acoustics.hybrid.synthesis.HybridResult` per pair.
    num_delays : int
        The delay lines of the network."""


class Binaural(_HybridMethod):
    """Binaural room impulse responses: image sources, optional edge diffraction, a directional FDN.

    The early field is the image sources up to ``image_source_order`` (and, with ``diffraction``, edge
    diffraction). The last order drives a feedback delay network of 96 lines, one per direction, whose decay
    follows a target (Wendt et al. 2014), the count Kirsch et al. (2023) use offline. Every early path and
    delay line is filtered with the HRIR measured nearest its direction. Needs Python >= 3.11 and a closed
    room. Responses are ears left then right, stored under ``'binaural'``; ``generation["hrtf"]`` names the set.

    Parameters
    ----------
    reference : str, dict, Mono or None
        A package-scale mono response per pair for the decay target and the late envelope: a method's name on
        the IR graph (``"mono"``, ``"data"``), a dict as :meth:`Scene.simulate` returns, or a
        :class:`~compas_acoustics.simulation_methods.Mono` to run first.
    hrtf : str or path, optional
        ``"kemar"`` (None), the KEMAR set of the HRTF library, or a SOFA file.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> ir = next(iter(scene.simulate(ca.Binaural(tail="eyring")).values()))  # doctest: +SKIP
    >>> ir.n_channels, ir.generation["hrtf"]  # doctest: +SKIP
    (2, 'KEMAR_FreeFieldComp_48kHz.sofa')

    With a ray-traced reference the decay is fitted to a response you already have.

    >>> reference = scene.simulate(ca.Mono(ray_tracing=True))  # doctest: +SKIP
    >>> responses = scene.simulate(ca.Binaural(reference="mono"))  # doctest: +SKIP
    """

    name = "binaural"
    output = "binaural"
    num_delays = MIN_DELAYS

    def __init__(self, reference=None, hrtf=None, *, tail=None, image_source_order=3, n_slopes=1, decay_db=30.0,
                 duration=None, directional_absorption=True, late_envelope=None, diffraction=False, air_absorption=None,
                 seed=True, normalize=True, air=None, scattering=False):
        hrtf = "kemar" if hrtf is None else hrtf
        if hrtf != "kemar" and not Path(hrtf).is_file():
            raise ValueError(f"hrtf must be 'kemar' or a SOFA file, not {hrtf!r}")
        super().__init__(reference, tail, image_source_order, n_slopes, decay_db, duration, directional_absorption,
                         late_envelope, diffraction, air_absorption, seed, normalize, air, scattering)
        self.hrtf = hrtf
        self._hrtf_set = None
        self._hrtf_loaded = None

    def _response(self, core, key, source, receiver, generation):
        from compas_acoustics.hybrid.binaural import DEFAULT_HRTF
        from compas_acoustics.hybrid.binaural import HrtfSet
        from compas_acoustics.hybrid.binaural import render_binaural

        path = DEFAULT_HRTF if self.hrtf == "kemar" else Path(self.hrtf)
        # loaded once per run, again only when the set or the rate changed since
        if self._hrtf_loaded != (path, int(core.fs)):
            self._hrtf_set, self._hrtf_loaded = HrtfSet.from_sofa(path, int(core.fs)), (path, int(core.fs))
        result = self.results[key] = render_binaural(core, self._hrtf_set)
        return ImpulseResponse(name=key, source=source, receiver=receiver, ir=result.rir, sampling_rate=int(core.fs),
                               normalized=self.normalize, generation=dict(generation, hrtf=path.name))


class Ambisonics(_HybridMethod):
    """Ambisonic room impulse responses: image sources, optional edge diffraction, a directional FDN.

    As :class:`Binaural`, with every early path encoded at its direction and every delay line projected from
    its direction at ``sh_order``. Responses are ACN in ``normalization``, stored under ``'ambisonics'``.

    Parameters
    ----------
    reference : str, dict, Mono or None
        As :class:`Binaural`.
    sh_order : int, default=1
        The network has 96 delay lines up to third order (Kirsch et al. 2023), above that the smallest
        equal-area grid supporting the order. The input gain ``sqrt(num_delays)`` offsets the output's
        ``1 / sqrt(num_delays)``, so the late level does not depend on the count.
    normalization : {"SN3D", "N3D"}, default="SN3D"
        ACN channel order, W = 1 for a unit plane wave.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> ir = next(iter(scene.simulate(ca.Ambisonics(tail="eyring", sh_order=2)).values()))  # doctest: +SKIP
    >>> ir.n_channels, ir.ambisonic_order  # doctest: +SKIP
    (9, 2)
    """

    name = "ambisonics"
    output = "ambisonic"

    def __init__(self, reference=None, sh_order=1, normalization="SN3D", *, tail=None, image_source_order=3, n_slopes=1,
                 decay_db=30.0, duration=None, directional_absorption=True, late_envelope=None, diffraction=False,
                 air_absorption=None, seed=True, normalize=True, air=None, scattering=False):
        if normalization not in ("SN3D", "N3D"):
            raise ValueError(f"normalization must be 'SN3D' or 'N3D', not {normalization!r}")
        if sh_order < 0:
            raise ValueError("sh_order must be non-negative")
        super().__init__(reference, tail, image_source_order, n_slopes, decay_db, duration, directional_absorption,
                         late_envelope, diffraction, air_absorption, seed, normalize, air, scattering)
        self.sh_order = int(sh_order)
        self.normalization = normalization

    @property
    def num_delays(self) -> int:
        """96 up to third order, more above (:func:`~compas_acoustics.hybrid.ambisonics.default_num_delays`)."""
        return default_num_delays(self.sh_order)

    def _response(self, core, key, source, receiver, generation):
        result = self.results[key] = render_ambisonics(core, self.sh_order, self.normalization)
        return ImpulseResponse(name=key, source=source, receiver=receiver, ir=result.rir, sampling_rate=int(core.fs),
                               normalized=self.normalize, generation=dict(generation, sh_order=self.sh_order),
                               ambisonic_order=self.sh_order, normalization=self.normalization.lower(), channel_order="acn")


for _method in (Binaural, Ambisonics):
    _method.__doc__ = _method.__doc__.replace("\n\n    Examples\n", _ENGINE_PARAMETERS + "\n\n    Examples\n", 1)
del _method
