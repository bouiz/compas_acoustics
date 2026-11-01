A simple simulation in a shoebox room

A 5 × 4 × 3 m room with a different material on its floor, walls and ceiling, two sources and two receivers,
simulated with image sources and ray tracing. The definition reads the reverberation time T30 of one source and
receiver and plots it per octave band. It builds the same room as the Python version, wall for wall.

#. **Room.** A room drawn in Rhino would come in through a Mesh or Brep parameter; here Box 2Pt makes one from
   (0, 0, 0) to (5, 4, 3). Deconstruct Brep splits it into its six faces, List Item puts them in the
   order floor, right, front, left and back wall, ceiling, and Mesh Brep turns each into a mesh, the geometry an
   acoustic object takes.
#. **Materials.** Four :ref:`Construct Material <gh-CA_Construct_Material>` components each take seven absorption
   coefficients (125 Hz to 8 kHz) and seven scattering coefficients, or one for every band. Merge lists one material
   per wall, in the order of the walls. A material from the library would come from
   :ref:`Material Library <gh-CA_Material_Library>` instead.
#. **Acoustic objects.** :ref:`Acoustic Object <gh-CA_Acoustic_Object>` pairs each wall with its material and name.
   Its role is *boundary*: the walls enclose the room.
#. **Sources and receivers.** Two :ref:`Source <gh-CA_Source>` and two :ref:`Receiver <gh-CA_Receiver>` components,
   each with a position and a name. Both face +x, their default.
#. **Scene and simulation.** :ref:`Acoustic Scene <gh-CA_Model>` collects the walls, sources and receivers and reports
   the room's volume (60 m³) and area (94 m²). :ref:`Simulate Mono <gh-CA_Simulate_Mono>` traces 10000 rays with
   image sources up to order 3 and air absorption; the seed 42 makes the run repeatable. It runs in the background,
   in the worker.
#. **Analysis.** :ref:`Deconstruct IR Graph <gh-CA_Deconstruct_IR_Graph>` picks the response from source_2 to
   receiver_2 (its Source and Receiver dropdowns), :ref:`Analyse Impulse Response <gh-CA_Analyse_IR>` derives the
   room acoustic parameters, :ref:`Deconstruct Parameter <gh-CA_Deconstruct_IR>` selects T30, and
   :ref:`Plot 2D <gh-CA_Plot_2D>` draws it.

The values differ slightly from the Python version's. Ray tracing is random, and the order in which each wall lists
its corners changes which rays the seed draws; the differences are of the same size as those between two seeds.
