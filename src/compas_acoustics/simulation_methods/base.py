"""Simulation methods: interchangeable ways to compute a scene's impulse responses.

Every method takes the same :class:`SimulationContext` and returns one
:class:`~compas_acoustics.ir.ImpulseResponse` per source-receiver pair, keyed
``'<receiver guid>_<source guid>'``. :meth:`Scene.simulate` stores them in the
scene's IR graph under the method's :attr:`~SimulationMethod.attribute`, its
:attr:`~SimulationMethod.name` unless it renders several formats.

Every method's responses meet one contract, so that they can be compared,
mixed, analysed and auralised alike:

1. the **package scale**: free-field direct sound ``1 / (4 pi r)`` on its
   geometric sample, recorded as ``generation['scale'] == 'package'``;
2. the context's sample rate;
3. ``generation['method'] == method.name``, the settings, and
   ``generation['valid_range_hz']``: ``(low, high)`` in Hz where the method is
   trustworthy, or ``None`` for the full band;
4. a stated channel convention: mono, ``ambisonic_order``,
   ``normalization`` and ``channel_order``, or binaural (two channels, ears
   left then right, ``generation['output'] == 'binaural'``);
5. peak-normalized views by default, the raw level through ``denormalized()``.
"""

from __future__ import annotations

import numbers
import warnings
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import ClassVar

import numpy as np

from compas_acoustics.room.geometry import RoomGeometry

__all__ = ['SimulationContext', 'SimulationMethod', 'image_source_search_seconds', 'DEFAULT_SEED']

#: The seed ``seed=True``, every method's default, stands for
DEFAULT_SEED = 42


def _seed(seed):
    """A method's ``seed`` as the int to seed with, or None for no seed: True is :data:`DEFAULT_SEED`, False or None none."""
    if seed is None or isinstance(seed, (bool, np.bool_)):
        return DEFAULT_SEED if seed else None
    if isinstance(seed, numbers.Integral):
        return int(seed)
    raise ValueError(f"seed must be True, False or an int, not {seed!r}")


def image_source_search_seconds(n_faces: int, order: int) -> float:
    """A rough estimate of pyroomacoustics' image source search time on a mesh room, in seconds.

    The search is exponential in the order, roughly a factor ``n_faces / 2.2`` per
    order (measured with pyroomacoustics 0.10.1 on rooms of 6 to 42 faces: 0.7 s at
    order 6 with 18 faces, 120 s with 42). Machine-dependent; for warnings only.
    """
    return 1e-6 * n_faces ** 3.4 * (n_faces / 2.2) ** (order - 4)


def _warn_if_the_search_is_slow(geometry, order: int, stacklevel: int = 3) -> None:
    seconds = image_source_search_seconds(len(geometry.faces), order)
    if seconds > 30.0:
        when = "minutes" if seconds > 120.0 else "tens of seconds"
        warnings.warn(
            f"The image source search over {len(geometry.faces)} faces up to order {order} may take {when} or more "
            f"(it grows about {len(geometry.faces) / 2.2:.0f}x per order). Lower image_source_order (ray tracing "
            "supplies the later reflections), or simplify the geometry (coplanar faces are merged already; curved surfaces need fewer faces).",
            UserWarning,
            stacklevel=stacklevel,
        )


@dataclass
class SimulationContext:
    """What every method sees, built once per :meth:`Scene.simulate`."""

    scene: object
    geometry: object  # RoomGeometry
    sources: list  # inside the room
    receivers: list  # inside the room
    sampling_rate: int
    outside: list = field(default_factory=list)  # names of sources and receivers outside the room

    @classmethod
    def from_scene(cls, scene) -> "SimulationContext":
        """The scene as every method sees it: its room, at its sampling rate, and the sources and receivers inside it."""
        geometry = RoomGeometry.from_scene(scene)
        sources, receivers, outside = [], [], []
        for kind, items, inside in (("Source", scene.sources, sources), ("Receiver", scene.receivers, receivers)):
            for item in items:
                if geometry.is_inside(item.position):
                    inside.append(item)
                else:
                    outside.append(item.name)
                    warnings.warn(f"{kind} '{item.name}' is outside the room and will not be simulated.", UserWarning, stacklevel=3)
        for source in sources:
            for receiver in receivers:
                # The free-field spreading 1 / (4 pi r) is undefined there; pyroomacoustics returns NaN
                if np.linalg.norm(np.subtract(source.position, receiver.position)) < 1e-3:
                    raise ValueError(f"Source '{source.name}' and receiver '{receiver.name}' are at the same position; move one of them.")
        return cls(scene=scene, geometry=geometry, sources=sources, receivers=receivers,
                   sampling_rate=scene.sampling_rate, outside=outside)

    @staticmethod
    def key(source, receiver) -> str:
        return f"{receiver.guid}_{source.guid}"


class SimulationMethod(ABC):
    """A way to compute impulse responses. See the module docstring for the output contract."""

    #: ``generation['method']``, and the IR graph attribute the results are stored under unless :attr:`attribute` says otherwise
    name: ClassVar[str] = ""
    #: ``'mono'``, ``'ambisonic'``, ``'binaural'``, or a list of formats rendered from one run
    output = "mono"

    @property
    def attribute(self) -> str:
        """The IR graph attribute :meth:`Scene.simulate` stores :meth:`compute`'s responses under: :attr:`name`."""
        return self.name

    @property
    def extra_responses(self) -> dict:
        """Responses of the last :meth:`compute` in further formats, ``{attribute: responses}``: none."""
        return {}

    @abstractmethod
    def compute(self, context: SimulationContext) -> dict:
        """One response per source-receiver pair, keyed ``'<receiver guid>_<source guid>'``."""
