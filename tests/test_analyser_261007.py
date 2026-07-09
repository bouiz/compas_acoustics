import compas_acoustics as ca
import numpy as np
from compas.geometry import Vector


# 1. Create sources and receivers
source_1 = ca.Source(name="source1", x=4, y=3, z=1.5, directivity="hypercardioid", direction=Vector(0, 1, 0))

source_2 = ca.Source(name="source_2", x=3.0, y=2.0, z=1.5, directivity="hypercardioid", direction=Vector(1, 0, 1))
receiver_1 = ca.Receiver(name="receiver_1", x=2.0, y=2.0, z=1.6, direction=Vector(1, 0, 1))
receiver_2 = ca.Receiver(name="receiver_2", x=1.0, y=1.0, z=1.1, direction=Vector(1, 1, 1))

# 2. Create a shoebox room with concrete material
concrete = ca.AcousticMaterial(name='concrete', absorption_coefficients=[0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05], scattering_coefficients=[0.01, 0.01, 0.03, 0.05, 0.1, 0.1, 0.1])
# brickwall = ca.AcousticMaterial(name='brickwall', absorption_coefficients=[0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04], scattering_coefficients=[0.01, 0.01, 0.03, 0.05, 0.2, 0.25, 0.3])
room = ca.AcousticObjectsLibrary.shoebox_room(length=5, width=4, height=3, material=concrete)

# 3. Create a scene and add the room, sources, and receivers
scene = ca.Scene(name="Test Scene", description="This is a test scene.")

scene.add_acoustic_object(room)
scene.add_source(source_1)
scene.add_source(source_2)
scene.add_receiver(receiver_1)
scene.add_receiver(receiver_2)

# 4. Run simulation and analyze the results
simulator = ca.Simulator(scene, simulation_platform='pyroomacoustics')
rirs = simulator.rir_from_ray_tracing(n_rays=10000, air_absorption=True, random_seed=7)  # rirs is a dict

irs = [ir.ir for ir in rirs.values()]

analyser = ca.AcousticAnalyser(irs[0])
results = analyser.analyse()
data = np.transpose(results['data'])

edt = []
t20 = []
t30 = []
c50 = []
c80 = []
d50 = []
drr = []
ts = []
sti = []

for result in data[:-1]:
    edt.append(result[0])
    t20.append(result[1])
    t30.append(result[2])
    c50.append(result[3])
    c80.append(result[4])
    d50.append(result[5])
    drr.append(result[6])
    ts.append(result[7])
    sti.append(result[8])

print (data)