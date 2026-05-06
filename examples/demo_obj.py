import os
import compas_acoustics as ca


########### create a room from an OBJ file
room_obj_path = os.path.join(os.path.dirname(__file__), 'sample_room.obj')
room = ca.AcousticObject.acoustic_object_from_obj(room_obj_path, name="sample_room")

table_obj_path = os.path.join(os.path.dirname(__file__), 'sample_table.obj')
table = ca.AcousticObject.acoustic_object_from_obj(table_obj_path, name="sample_table")


########### create a source and receivers
source = ca.Source(x=1.5, y=3.0, z=1.5, name="source_1")
receiver_1 = ca.Receiver(x=2.0, y=2.0, z=1.0, name="receiver_1")
# receiver_2 = ca.Receiver(x=6.0, y=4.0, z=1.0, name="receiver_2")


########### create a scene
scene = ca.Scene(name="Demo Scene", description="This is a demo scene.")


########### add the room, source, receiver, and materials to the scene
scene.add_acoustic_object(room)
scene.add_acoustic_object(table)
scene.add_source(source)
scene.add_receiver(receiver_1)
# scene.add_receiver(receiver_2)


########### visualize the scene and its surface groups
scene.display_surface_groups()


########### what are the scene's surface groups?
# scene.print_scene_surface_groups()


########### create acoustic materials
concrete = ca.AcousticMaterial(name='concrete', absorption_coefficients=[0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05], scattering_coefficients=[0.01, 0.01, 0.03, 0.05, 0.1, 0.1, 0.1])
plasterboard = ca.AcousticMaterial(name='plasterboard', absorption_coefficients=[0.15, 0.1, 0.06, 0.04, 0.04, 0.05, 0.05], scattering_coefficients=[0.01, 0.01, 0.03, 0.05, 0.2, 0.1, 0.1])
acoustic_plaster = ca.AcousticMaterial(name='acoustic_plaster', absorption_coefficients=[0.17, 0.36, 0.66, 0.65, 0.62, 0.68, 0.80], scattering_coefficients=0.2)
glass = ca.AcousticMaterial(name='glass', absorption_coefficients=[0.1, 0.07, 0.05, 0.03, 0.02, 0.02, 0.02], scattering_coefficients=0.05)
plywood = ca.AcousticMaterial(name='plywood', absorption_coefficients=[0.18, 0.12, 0.1, 0.09, 0.08, 0.07, 0.07], scattering_coefficients=[0.05, 0.3, 0.03, 0.5, 0.6, 0.7, 0.7])


########### create a dictionary mapping surface groups to acoustic materials
scene_materials = {
    'floor': concrete,
    'plasterboard': plasterboard,
    'glass': glass,
    'ceiling': concrete,
    'wood': plywood
}
scene.acoustic_materials = scene_materials


########### initialize the simulation
scene.simulation_platform = 'pyroomacoustics'


########### run an Image Source Method simulation
# rirs = scene.rir_from_image_source_method()


########### run a ray tracing simulation
rirs = scene.rir_from_ray_tracing(n_rays=10000)


########### analyze the RIRs
for rir in rirs:
    analyser = ca.AcousticAnalyser(rir[0])
    results = analyser.analyse()


    ########### print the results
    analyser.print_table()


    ########### plot results
    analyser.plot(['T30', 'C50'])
