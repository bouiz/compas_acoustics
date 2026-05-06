import compas_acoustics as ca
from scipy.io.wavfile import write
import os


########### create a shoebox room
room = ca.AcousticObjectsLibrary.shoebox_room(length=5, width=4, height=3)


########### create a source and a receiver
source_1 = ca.Source(x=1.0, y=1.0, z=1.0, name="source_1")
source_2 = ca.Source(x=1.5, y=1.5, z=1.0, name="source_2")
receiver_1 = ca.Receiver(x=2.0, y=2.0, z=1.5, name="receiver_1")
receiver_2 = ca.Receiver(x=2.5, y=2.0, z=1.5, name="receiver_2")


########### create a scene
scene = ca.Scene(name="Demo Scene", description="This is a demo scene.")


########### add the room, source, and receiver to the scene
scene.add_acoustic_object(room)
scene.add_source(source_1)
# scene.add_source(source_2)
scene.add_receiver(receiver_1)
# scene.add_receiver(receiver_2)
########### what are the scene's surface groups?
# scene.print_scene_surface_groups()


########### visualize the scene and its surface groups
# scene.display_surface_groups()


########### create acoustic materials
concrete = ca.AcousticMaterial(name='concrete', absorption_coefficients=[0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05], scattering_coefficients=[0.01, 0.01, 0.03, 0.05, 0.1, 0.1, 0.1])
marble = ca.AcousticMaterial(name='marble', absorption_coefficients=[0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02], scattering_coefficients=[0.01, 0.01, 0.03, 0.05, 0.1, 0.1, 0.1])
brickwall = ca.AcousticMaterial(name='brickwall', absorption_coefficients=[0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04], scattering_coefficients=[0.01, 0.01, 0.03, 0.05, 0.2, 0.25, 0.3])
carpet = ca.AcousticMaterial(name='carpet', absorption_coefficients=[0.07, 0.31, 0.49, 0.81, 0.66, 0.54, 0.48], scattering_coefficients=0.2)
# acoustic_plaster = ca.AcousticMaterial(name='acoustic_plaster', absorption_coefficients=[0.17, 0.36, 0.66, 0.65, 0.62, 0.68, 0.80], scattering_coefficients=0.2)

########### create a dictionary mapping surface groups to acoustic materials
scene_materials = {
    'floor': carpet,
    'left_wall': brickwall,
    'front_wall': brickwall,
    'right_wall': brickwall,
    'back_wall': brickwall,
    'ceiling': concrete,
}
scene.acoustic_materials = scene_materials


########### initialize the simulation
scene.simulation_platform = 'pyroomacoustics'


########### run an Image Source Method simulation
# rir = scene.rir_from_image_source_method()


########### run a ray tracing simulation
rir = scene.rir_from_ray_tracing(n_rays=10000, air_absorption=True)

# export_path = os.path.join(os.path.dirname(__file__), 'rir_16k.wav')
# write(export_path, scene.sampling_rate, rir[0][0])


########### analyze the RIR
analyser = ca.AcousticAnalyser(rir[0][0])
results = analyser.analyse()


# ########### print the results
analyser.print_table()


# ########### plot results
# analyser.plot_rir()
analyser.plot_frequency_response(smooth=3)
# analyser.plot('T30')