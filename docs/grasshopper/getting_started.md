# Getting started in Grasshopper

After [installing the plugin](../installation/grasshopper.md), the components are in the **COMPAS Acoustics** tab,
in panels from left to right in the order a definition uses them. [Concepts](../concepts.md) explains the terms.

## How a definition flows

1. **Elements.** Turn Rhino geometry into acoustic objects: pick a material from the
   [Material Library](components/CA_Material_Library.md) or make one with
   [Construct Material](components/CA_Construct_Material.md), then give each mesh (or closed polyline) its
   material in [Acoustic Object](components/CA_Acoustic_Object.md); turn Breps into meshes with Mesh Brep first. Place a [Source](components/CA_Source.md) and a
   [Receiver](components/CA_Receiver.md).
2. **Model.** Collect objects, sources and receivers in an [Acoustic Scene](components/CA_Model.md).
3. **Simulation.** Run [Simulate Mono](components/CA_Simulate_Mono.md),
   [Simulate Binaural](components/CA_Simulate_Binaural.md) or [Simulate Ambisonics](components/CA_Simulate_Ambisonics.md).
   They return a graph of impulse responses, one per source and receiver; pick one with
   [Deconstruct IR Graph](components/CA_Deconstruct_IR_Graph.md).
4. **Analysis.** [Analyse Impulse Response](components/CA_Analyse_IR.md) gives the room acoustic parameters, and
   [Plot 2D](components/CA_Plot_2D.md) draws them on the canvas.
5. **Auralisation.** [Auralise](components/CA_Auralise.md) convolves a recording with the responses;
   [Audio Player](components/CA_Audio_Player.md) plays it.
6. **Display.** Show reflection paths, rays, a billiard of sound particles or an energy map in the Rhino viewport.

[A simple simulation in a shoebox room](examples/simple_simulation_shoebox.md) builds steps 1 to 4 for a shoebox
room. Every component is described in [Components](components/index.md).

## Using the components

Choices sit on the component bodies (switches, dropdowns, buttons); a wire into the same input wins over the
control. The long runs (simulations, analysis, auralisation, rays) work in the background: the canvas stays
responsive, the message shows the time and the worker's stage, and Cancel stops the run. Material Library opens a
searchable picker on a click of its body; Plot 2D draws its charts on the canvas; the Audio Player shows the signal
(an auralisation, a file, or an impulse response from Deconstruct IR Graph) and plays, pauses, seeks and sets the
volume.

## The worker

The components compute nothing themselves: they send the work to the *worker*, the Python 3.12 environment the
installer created. It starts with the first component that needs it, a few seconds (half a minute the first time),
and stops with Rhino.

The [Worker](components/CA_Worker.md) component (Util panel) shows its state, for example
`compas_acoustics 0.1.0, schema ok`. Every component's right-click menu has **Open worker log**, which opens the
worker's log file: read it when a component turns red and its message is not enough.

Messages you may see:

`No worker: run the compas_acoustics installer.`
:   The worker was never created on this computer. Run the installer.

`... is not this version's ...: run the installer again.` / `schema mismatch: run the installer again`
:   The plugin and the worker come from different releases. Run the installer of the plugin's release.

`The worker does not answer: ... Recompute to start it again.`
:   The worker stopped (or was stopped). Recompute the component.

More in [Troubleshooting](../troubleshooting.md).
