from compas_acoustics import AcousticMaterial, Source, Receiver, AcousticObject, AcousticObjectsLibrary, AcousticAnalyser
from compas_acoustics.materials_library import MaterialsLibrary
from compas_acoustics.defaults import Defaults
from compas.data import json_dumps, json_dump, json_loads, json_load
from compas.files import OBJ
from compas_viewer import Viewer
from compas_viewer.scene import Tag
from compas.colors import Color
from random import random
from compas_acoustics import Scene

import matplotlib.pyplot as plt


# viewer = Viewer()

# analyser = Analyser()
# print(analyser.name)
# print(analyser.guid)



# aa = AcousticObject.acoustic_object_from_obj('/Users/achilleasxydis/Desktop/gg3.obj')
aa = AcousticObjectsLibrary.shoebox_room(length=5, width=4, height=3)
# print(aa.surface_groups)
# print(list(aa.acoustic_geometry.faces()))
aa_mesh = aa.acoustic_geometry
# print(aa_mesh.face_attribute(0, 'surface_group'))
# for fkey in aa_mesh.faces():
#     print(fkey, len(aa_mesh.face_vertices(fkey)))


scene = Scene(name="Test Scene", description="This is a test scene.")
scene.add_acoustic_object(aa)
# scene.add_acoustic_object(bb)
# # scene.add_acoustic_objects([aa, bb])
# print(scene.scene_surface_groups)

concrete = AcousticMaterial(name='concrete', absorption_coefficients=[0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01], scattering_coefficients=0.2)
brickwall = MaterialsLibrary().material_from_name('test_material')
carpet = AcousticMaterial(name='carpet', absorption_coefficients=[0.01, 0.05, 0.1, 0.15, 0.3, 0.4, 0.4], scattering_coefficients=0.1)
# hzzz = MaterialsLibrary().material_from_name('test_material')

# carpet.add_to_library()


scene_materials = {
   'floor' : carpet,
   'left_wall' : brickwall,
   'front_wall' : brickwall,
   'right_wall' : brickwall,
   'back_wall' : brickwall,
   'ceiling' : concrete,
}

scene.acoustic_materials = scene_materials
sim = scene.simulation

# print(sim.get_volume())


# # print("Simulation object:", sim)
src = [0.0, 0.0, 1.2]
mic = [0.5, 2.0, 1.3]
sim.add_source(src)
sim.add_microphone(mic)

# fig, axes = sim.plot()
# print(fig)
# plt.show(block=True)

# print("MIC: ", sim.is_inside(mic))
# print("SRC: ", sim.is_inside(src))

# compute the rir
sim.set_ray_tracing(n_rays=10000)
sim.image_source_model()
sim.compute_rir()
rir = sim.rir[0][0]
sim.plot_rir()
plt.show()


analyser = AcousticAnalyser(rir)
analyser.analyse()
analyser.print_table()
analyser.plot('T30')

# # # show the room
# sim.plot(img_order=1)
# plt.show()




###### FACE COLORS BASED ON SURFACE GROUPS ######

# group_colors = aa.surface_group_colors
# facecolor = aa.face_colors

# for fkey in aa_mesh.faces():
#     f_centroid = aa_mesh.face_centroid(fkey)
#     surface_group = aa_mesh.face_attribute(fkey, 'surface_group')
#     tag = Tag(surface_group, f_centroid, color=Color.black())
#     viewer.scene.add(tag, name=f"tag_{fkey}")

# viewer.scene.add(aa_mesh, name="mesh1", show_points=True, facecolor=facecolor)

# viewer.show()

##################################################



# source = Source(name='source_1', x=0, y=0, z=1)
# receiver = Receiver(name='seceiver_1', x=5, y=5, z=1)



# print(material.scattering_coefficients)
# material.scattering_coefficients = 0.2
# material.scattering_coefficients_from_depth(0.2)
# print(material.scattering_coefficients)

# material.to_json('/Users/achilleasxydis/PROJECTS/python/compas_acoustics/src/compas_acoustics/materials_library/test_material2.json', compact=True, minimal=True )



# material.scattering_coefficients = [0.2]

# material.to_json('/Users/achilleasxydis/PROJECTS/python/compas_acoustics/src/compas_acoustics/materials_library/test_material_1.json', compact=True, minimal=True )

# data = json_dumps(material)
# json_dump(material, fp='/Users/achilleasxydis/PROJECTS/python/compas_acoustics/src/compas_acoustics/materials_library/test_material_2.json', compact=True, minimal=True )
# print("Serialized Data:", data)


# m = json_load('/Users/achilleasxydis/PROJECTS/python/compas_acoustics/src/compas_acoustics/materials_library/test_material.json')

# print(m)
