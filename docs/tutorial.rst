********************************************************************************
Tutorial
********************************************************************************

This tutorial builds a room, simulates its impulse responses, analyses them and
listens to the result. Every step uses the classes of :mod:`compas_acoustics`.


Build a scene
=============

A :class:`~compas_acoustics.Scene` holds the room's surfaces, the sources and
the receivers. A shoebox room is six surfaces, each an
:class:`~compas_acoustics.AcousticObject` with its own material.

.. code-block:: python

    import compas_acoustics as ca

    scene = ca.Scene("room")
    scene.add_source(ca.Source("talker", 2, 2, 1.6))
    scene.add_receiver(ca.Receiver("listener", 6, 4, 1.2, direction=[-1, -0.5, 0]))

A room drawn in Rhino can be read from an OBJ file instead, one object per OBJ
group (Rhino layer): ``ca.AcousticObject.from_obj(path, materials={...})``.
``examples/obj.py`` does this for a room with a table in it.


Materials
=========

Materials have absorption and scattering coefficients in seven octave bands,
125 Hz to 8 kHz. Take them from the library, or define your own.

.. code-block:: python

    library = ca.MaterialsLibrary()
    carpet = library.material_from_name("carpet")
    plaster = ca.AcousticMaterial("plaster", absorption_coefficients=0.05, scattering_coefficients=0.1)

    room = ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials={
        "floor": carpet,
        "ceiling": plaster,
        "left_wall": plaster,
        "right_wall": plaster,
        "front_wall": plaster,
        "back_wall": plaster,
    })
    scene.add_acoustic_objects(room)


Simulate
========

Every simulation method runs through ``scene.simulate``, which returns one
:class:`~compas_acoustics.ImpulseResponse` per source-receiver pair.
:class:`~compas_acoustics.Mono` uses image sources and ray tracing;
:class:`~compas_acoustics.Binaural` and :class:`~compas_acoustics.Ambisonics`
use the hybrid engine and need Python 3.11 or newer.

.. code-block:: python

    mono = scene.simulate(ca.Mono(ray_tracing=True, n_rays=5000))
    binaural = scene.simulate(ca.Binaural())
    ambisonics = scene.simulate(ca.Ambisonics(sh_order=1))


Analyse
=======

:class:`~compas_acoustics.Analyser` derives the room acoustic parameters of
ISO 3382 from a response: reverberation times, clarity, definition and more,
per octave band.

.. code-block:: python

    (ir,) = mono.values()
    analyser = ca.Analyser(ir)
    results = analyser.analyse()
    analyser.print_table()


Auralise
========

:class:`~compas_acoustics.Auraliser` convolves the responses with an anechoic
recording, in the format of the responses: here binaural, for headphones.

.. code-block:: python

    speech = ca.Defaults().ANECHOIC_SOUNDFILES_LIBRARY_PATH + "/female_speech.wav"
    signal = ca.Auraliser("room", list(binaural.values()), anechoic_sounds=[speech]).convolve()

Ambisonic responses are convolved the same way and give an ambisonic (AmbiX)
signal, to save as a multichannel file for a decoder elsewhere.


Grasshopper
===========

The Grasshopper plugin has a component for each of these steps, in the tabs
``01 Model`` to ``06 Util`` of the COMPAS Acoustics category. Install it with the
installer bundle (see :doc:`installation`); the components run everything in a
Python worker that starts with the first component that needs it.
