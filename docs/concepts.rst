********************************************************************************
Concepts
********************************************************************************

This page explains the ideas that the Python package and the Grasshopper components share. Both tracks link here
instead of repeating them.


The model
=========

A **scene** holds everything a simulation needs:

Acoustic objects
    The surfaces of the room and the obstacles in it: meshes from Rhino, OBJ files or COMPAS geometry. Each one has a
    material. A *boundary* object encloses the room; an *obstacle*, such as a table, stands inside it.

Materials
    How a surface treats the sound that hits it, in seven octave bands from 125 Hz to 8 kHz. The **absorption
    coefficient** is the share of the sound energy the surface does not reflect (0: a hard wall, 1: an open window).
    The **scattering coefficient** is the share of the reflected energy it sends in random directions instead of as a
    mirror would (0: smooth, 1: very rough). The materials library has 95 of them in categories.

Sources and receivers
    Where the sound is made and where it is heard: a point with a direction. A source can have a sound power level
    and a sound file; a receiver is a listener's position and the way they face.


The impulse response
====================

A simulation computes one **impulse response** (IR) for each source and receiver: what the receiver hears when the
source makes a very short click. It holds everything about the path between them: the direct sound, the early
reflections from the nearest surfaces, and the late reverberation as the reflections pile up and die away. Every room
acoustic parameter below is read from it, and convolving it with a recording (*auralisation*) lets you hear the
recording as if it were played in the room.


Simulation methods
==================

All three methods run through ``scene.simulate`` (in Grasshopper, the Simulate components) and return one impulse
response per source and receiver.

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Method
     - How it works
     - Use it for
   * - ``Mono``
     - Image sources for the early reflections and ray tracing for the rest, with pyroomacoustics. One channel.
     - Room acoustic parameters, quick design comparisons. Runs on Python 3.9 and newer, also in Rhino 8.
   * - ``Binaural``
     - A hybrid engine of image sources, edge diffraction and a directional feedback delay network, rendered for two
       ears with a measured head (KEMAR).
     - Listening on headphones. Needs Python 3.11 or newer (the worker has it).
   * - ``Ambisonics``
     - The same hybrid engine, rendered as an ambisonic signal (ACN/SN3D, AmbiX) of a chosen order.
     - Lateral energy (LF, LFC), and playback on a loudspeaker array through an ambisonic decoder elsewhere.


Room acoustic parameters
========================

The analyser derives these parameters (after ISO 3382) from an impulse response, per octave band.

.. list-table::
   :header-rows: 1
   :widths: 12 58 30

   * - Parameter
     - What it means for a listener
     - Needs
   * - EDT
     - Early decay time: how reverberant the room *sounds*, from the first 10 dB of the decay. Seconds.
     -
   * - T20, T30
     - Reverberation time: how long a sound takes to fall by 60 dB, estimated from 20 or 30 dB of the decay. Seconds.
     -
   * - C50
     - Clarity for speech: early (first 50 ms) to late energy. Higher is clearer. dB.
     -
   * - C80
     - Clarity for music: early (first 80 ms) to late energy. dB.
     -
   * - D50
     - Definition: the share of the energy that arrives in the first 50 ms. Percent.
     -
   * - DRR
     - Direct-to-reverberant ratio: the direct sound against everything after it. dB.
     -
   * - Ts
     - Centre time: the centre of gravity of the energy in time. Lower is clearer. Milliseconds.
     -
   * - STI
     - Speech transmission index: speech intelligibility from 0 (bad) to 1 (excellent).
     -
   * - G
     - Strength: how loud the room makes a source, against the same source 10 m away in free field. dB.
     - A simulated response.
   * - SPL
     - Sound pressure level at the receiver from the source's level at 1 m, overall A-weighted. dB.
     - A simulated response.
   * - LF, LFC
     - Early lateral energy fraction: how much early sound arrives from the sides, which widens the apparent source.
     - An ambisonic response.


Auralisation
============

The auraliser convolves anechoic recordings (sound recorded without any room) with impulse responses and mixes the
sources. It keeps the format of the responses: a mono response gives mono, a binaural one gives a two-channel file for
headphones, an ambisonic one gives a multichannel file for an ambisonic decoder. There is no ambisonic-to-binaural
decoder: for headphones, simulate with ``Binaural``. The package bundles four anechoic recordings (see
:doc:`license`).
