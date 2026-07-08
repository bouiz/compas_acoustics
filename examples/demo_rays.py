"""Visualise the specular rays of a ray-tracing simulation.

This example traces specular rays from the sources of a shoebox room directly
into a spotlight cone (a direction + half angle) in a single call, then keeps a
single source and displays them, optionally truncated by reflection order or
sound travel time.
"""

from compas.geometry import Vector

import compas_acoustics as ca

########### create a shoebox room
room = ca.AcousticObjectsLibrary.shoebox_room(length=5, width=4, height=3)


########### create sources and receivers
source_1 = ca.Source(x=4.0, y=3.0, z=1.5, name="source_1")
source_2 = ca.Source(x=1.0, y=1.0, z=1.5, name="source_2")
receiver_1 = ca.Receiver(x=2.0, y=2.0, z=1.6, name="receiver_1")


########### create a scene
scene = ca.Scene(name="Ray Demo Scene", description="Specular ray visualisation.")
scene.add_acoustic_object(room)
scene.add_source(source_1)
scene.add_source(source_2)
scene.add_receiver(receiver_1)


########### assign acoustic materials
concrete = ca.AcousticMaterial(name="concrete", absorption_coefficients=[0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05])
scene.acoustic_materials = {
    "floor": concrete,
    "left_wall": concrete,
    "front_wall": concrete,
    "right_wall": concrete,
    "back_wall": concrete,
    "ceiling": concrete,
}


########### trace the rays directly into a spotlight cone
cone_axis = Vector(-1, -1, 0)
simulator = ca.Simulator(scene, simulation_platform="pyroomacoustics")
rays = simulator.trace_rays(n_rays=500, max_reflection_order=6, direction=cone_axis, angle=25)
print(f"Traced {len(rays)} rays into the 25 degree cone.")


########### keep only the rays from source_1
filtered = rays.filter(source=source_1)
print(f"{len(filtered)} rays emitted from source_1.")


########### display, truncated to the first 3 reflections (or use travel_time_ms=...)
filtered.display_in_viewer(scene=scene, reflection_order=3)
