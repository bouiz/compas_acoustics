A simple simulation in a shoebox room

A 5 × 4 × 3 m room with a different material on its floor, walls and ceiling, two sources and two receivers,
simulated with image sources and ray tracing. The definition reads the reverberation time T30 of one source and
receiver and plots it per octave band. It builds the same room as the Python version, wall for wall.

1. **Room.** A room drawn in Rhino would come in through a Mesh or Brep parameter; here Box 2Pt makes one from
   (0, 0, 0) to (5, 4, 3). Deconstruct Brep splits it into its six faces, List Item puts them in the
   order floor, right, front, left and back wall, ceiling, and Mesh Brep turns each into a mesh, the geometry an
   acoustic object takes.
2. **Materials.** Four [Construct Material](../components/CA_Construct_Material.md) components each take seven absorption
   coefficients (125 Hz to 8 kHz) and seven scattering coefficients, or one for every band. Merge lists one material
   per wall, in the order of the walls. A material from the library would come from
   [Material Library](../components/CA_Material_Library.md) instead.
3. **Acoustic objects.** [Acoustic Object](../components/CA_Acoustic_Object.md) pairs each wall with its material and name.
   Its role is *boundary*: the walls enclose the room.
4. **Sources and receivers.** Two [Source](../components/CA_Source.md) and two [Receiver](../components/CA_Receiver.md) components,
   each with a position and a name. Both face +x, their default.
5. **Scene and simulation.** [Acoustic Scene](../components/CA_Model.md) collects the walls, sources and receivers and reports
   the room's volume (60 m³) and area (94 m²). [Simulate Mono](../components/CA_Simulate_Mono.md) traces 10000 rays with
   image sources up to order 3 and air absorption; the seed 42 makes the run repeatable. It runs in the background,
   in the worker.
6. **Analysis.** [Deconstruct IR Graph](../components/CA_Deconstruct_IR_Graph.md) picks the response from source_2 to
   receiver_2 (its Source and Receiver dropdowns), [Analyse Impulse Response](../components/CA_Analyse_IR.md) derives the
   room acoustic parameters, [Deconstruct Parameter](../components/CA_Deconstruct_IR.md) selects T30, and
   [Plot 2D](../components/CA_Plot_2D.md) draws it.

The values differ slightly from the Python version's. Ray tracing is random, and the order in which each wall lists
its corners changes which rays the seed draws; the differences are of the same size as those between two seeds.
