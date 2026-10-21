"""Visualise the specular reflection paths of the image source method.

This example finds the image-source paths from the source to the receiver of a
shoebox room up to second order, the same images pyroomacoustics finds, and
displays the first and second order reflections.
"""

import compas_acoustics as ca


########### create a source and a receiver
source = ca.Source(x=1.5, y=1.5, z=1.6, name="source")
receiver = ca.Receiver(x=5.0, y=3.0, z=1.5, name="receiver")


########### create a scene
scene = ca.Scene(name="Specular Demo Scene", description="Specular reflection visualisation.")
scene.add_source(source)
scene.add_receiver(receiver)


########### assign acoustic materials
concrete = ca.AcousticMaterial(name="concrete", absorption_coefficients=[0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05])
room = ca.AcousticObjectsLibrary.shoebox_room(length=7, width=5, height=3, materials=concrete)
scene.add_acoustic_objects(room)


########### find the image-source paths up to second order
reflections = ca.SpecularReflections.from_scene(scene, 2)
for order in reflections.orders:
    paths = reflections.filter(order=order)
    print(f"order {order}: {len(paths)} paths, the first after {min(p.travel_time for p in paths):.2f} ms")


########### display the first and second order reflections
reflections.display_in_viewer(scene, order=[1, 2])
