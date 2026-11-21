"""Follow and stop long computations: simulations, energy maps, ray fields.

The computations report their stages as they go. Inside :func:`reporting` every report calls your function with
``(stage, done, of)``; raise from it, for instance :class:`Cancelled`, to stop the computation at that point.
Reports are per thread, so a host running several simulations in threads follows each one separately.

Examples
--------
>>> import compas_acoustics as ca
>>> from compas_acoustics.progress import reporting
>>> scene = ca.Scene("office")
>>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
>>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
>>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
>>> stages = []
>>> with reporting(lambda stage, done, of: stages.append(stage)):
...     _ = scene.simulate(ca.Mono(image_source_order=1, ray_tracing=False))
>>> stages[0]
'image sources'
"""

from compas_acoustics.worker.jobs import Cancelled
from compas_acoustics.worker.jobs import reporting

__all__ = ["Cancelled", "reporting"]
