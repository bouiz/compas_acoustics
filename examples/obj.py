"""A room from OBJ files.

A room and a table read from OBJ files, one acoustic object per OBJ group, the table as a free-standing
obstacle, simulated with image sources and ray tracing and analysed.
"""

import os
import compas_acoustics as ca


########### create a source and receivers
source = ca.Source(x=1.5, y=3.0, z=1.5, name="source_1")
receiver_1 = ca.Receiver(x=2.0, y=2.0, z=1.0, name="receiver_1")
# receiver_2 = ca.Receiver(x=6.0, y=4.0, z=1.0, name="receiver_2")


########### create a scene
scene = ca.Scene(name="Demo Scene", description="This is a demo scene.")


########### add the source and receiver to the scene
scene.add_source(source)
scene.add_receiver(receiver_1)
# scene.add_receiver(receiver_2)


########### what are the scene's surface groups?
# scene.print_scene_materials()


########### create acoustic materials
concrete = ca.AcousticMaterial(name='concrete', absorption_coefficients=[0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05], scattering_coefficients=[0.01, 0.01, 0.03, 0.05, 0.1, 0.1, 0.1])
plasterboard = ca.AcousticMaterial(name='plasterboard', absorption_coefficients=[0.15, 0.1, 0.06, 0.04, 0.04, 0.05, 0.05], scattering_coefficients=[0.01, 0.01, 0.03, 0.05, 0.2, 0.1, 0.1])
acoustic_plaster = ca.AcousticMaterial(name='acoustic_plaster', absorption_coefficients=[0.17, 0.36, 0.66, 0.65, 0.62, 0.68, 0.80], scattering_coefficients=0.2)
glass = ca.AcousticMaterial(name='glass', absorption_coefficients=[0.1, 0.07, 0.05, 0.03, 0.02, 0.02, 0.02], scattering_coefficients=0.05)
plywood = ca.AcousticMaterial(name='plywood', absorption_coefficients=[0.18, 0.12, 0.1, 0.09, 0.08, 0.07, 0.07], scattering_coefficients=[0.05, 0.3, 0.03, 0.5, 0.6, 0.7, 0.7])


########### read the room and the table from OBJ files, one acoustic object per OBJ group (Rhino layer)
room_obj_path = os.path.join(os.path.dirname(__file__), 'sample_room.obj')
room = ca.AcousticObject.from_obj(room_obj_path, materials={
    'floor': concrete,
    'plasterboard': plasterboard,
    'glass': glass,
    'ceiling': concrete,
}, name="sample_room")

# The table stands free in the room: an obstacle, reflecting from both sides
table_obj_path = os.path.join(os.path.dirname(__file__), 'sample_table.obj')
table = ca.AcousticObject.from_obj(table_obj_path, materials=plywood, name="sample_table")
ca.AcousticObject.assign_roles_by_group(table, {"wood": "obstacle"})

scene.add_acoustic_objects(room)
scene.add_acoustic_objects(table)


########### run a ray tracing simulation
rirs = scene.simulate(ca.Mono(ray_tracing=True, n_rays=10000, air_absorption=True))


########### analyze the RIR
analyser = ca.Analyser(next(iter(rirs.values())))
results = analyser.analyse()


########### print the results
analyser.print_table()


########## plot results
# analyser.plot_rir()
# analyser.plot_frequency_response(smooth=3)
analyser.plot('T30')
