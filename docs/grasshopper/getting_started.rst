********************************************************************************
Getting started in Grasshopper
********************************************************************************

After :doc:`installing the plugin </installation/grasshopper>`, the components are in the **COMPAS Acoustics** tab,
in panels from left to right in the order a definition uses them. :doc:`/concepts` explains the terms.


How a definition flows
======================

1. **Elements.** Turn Rhino geometry into acoustic objects: pick a material from the
   :ref:`Material Library <gh-CA_Material_Library>` or make one with
   :ref:`Construct Material <gh-CA_Construct_Material>`, then give each mesh or Brep its material in
   :ref:`Acoustic Object <gh-CA_Acoustic_Object>`. Place a :ref:`Source <gh-CA_Source>` and a
   :ref:`Receiver <gh-CA_Receiver>`.
2. **Model.** Collect objects, sources and receivers in an :ref:`Acoustic Scene <gh-CA_Model>`.
3. **Simulation.** Run :ref:`Simulate Mono <gh-CA_Simulate_Mono>`,
   :ref:`Simulate Binaural <gh-CA_Simulate_Binaural>` or :ref:`Simulate Ambisonics <gh-CA_Simulate_Ambisonics>`.
   They return a graph of impulse responses, one per source and receiver; pick one with
   :ref:`Deconstruct IR Graph <gh-CA_Deconstruct_IR_Graph>`.
4. **Analysis.** :ref:`Analyse Impulse Response <gh-CA_Analyse_IR>` gives the room acoustic parameters, and
   :ref:`Plot 2D <gh-CA_Plot_2D>` draws them on the canvas.
5. **Auralisation.** :ref:`Auralise <gh-CA_Auralise>` convolves a recording with the responses;
   :ref:`Audio Player <gh-CA_Audio_Player>` plays it.
6. **Display.** Show reflection paths, rays, a billiard of sound particles or an energy map in the Rhino viewport.

:doc:`examples/simple_simulation_shoebox` builds steps 1 to 4 for a shoebox room. Every component is described in
:doc:`components/index`.


Using the components
====================

Choices sit on the component bodies (switches, dropdowns, buttons); a wire into the same input wins over the
control. The long runs (simulations, analysis, auralisation, rays) work in the background: the canvas stays
responsive, the message shows the time and the worker's stage, and Cancel stops the run. Material Library opens a
searchable picker on a click of its body; Plot 2D draws its charts on the canvas; the Audio Player shows the signal
(an auralisation, a file, or an impulse response from Deconstruct IR Graph) and plays, pauses, seeks and sets the
volume.


The worker
==========

The components compute nothing themselves: they send the work to the *worker*, the Python 3.12 environment the
installer created. It starts with the first component that needs it, a few seconds (half a minute the first time),
and stops with Rhino.

The :ref:`Worker <gh-CA_Worker>` component (Util panel) shows its state, for example
``compas_acoustics 0.1.0, schema ok``. Every component's right-click menu has **Open worker log**, which opens the
worker's log file: read it when a component turns red and its message is not enough.

Messages you may see:

``No worker: run the compas_acoustics installer.``
    The worker was never created on this computer. Run the installer.
``... is not this version's ...: run the installer again.`` / ``schema mismatch: run the installer again``
    The plugin and the worker come from different releases. Run the installer of the plugin's release.
``The worker does not answer: ... Recompute to start it again.``
    The worker stopped (or was stopped). Recompute the component.

More in :doc:`/troubleshooting`.
