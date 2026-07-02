"""Visualise coefficient-driven scattered rays of a ray-tracing simulation.

At each reflection, extra scattered rays are spawned whose number and directions
are governed by the surface scattering coefficient: a mirror-like surface spawns
few rays clustered around the specular direction, while a highly diffusing surface
spawns many rays spread toward the surface normal.
"""

import compas_acoustics as ca

########### create a shoebox room
room = ca.AcousticObjectsLibrary.shoebox_room(length=5, width=4, height=3)


########### create a source and a receiver
source_1 = ca.Source(x=4.0, y=3.0, z=1.5, name="source_1")
receiver_1 = ca.Receiver(x=2.0, y=2.0, z=1.6, name="receiver_1")


########### create a scene
scene = ca.Scene(name="Scattered Ray Demo", description="Scattered ray visualisation.")
scene.add_acoustic_object(room)
scene.add_source(source_1)
scene.add_receiver(receiver_1)


########### assign materials with non-zero scattering coefficients
concrete = ca.AcousticMaterial(
    name="concrete",
    absorption_coefficients=[0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05]
    )

diffusing = ca.AcousticMaterial(
    name="diffuser",
    absorption_coefficients=[0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3],
    scattering_coefficients=[0.4, 0.5, 0.6, 0.7, 0.7, 0.8, 0.8],
)
scene.acoustic_materials = {
    "floor": concrete,
    "left_wall": diffusing,
    "front_wall": concrete,
    "right_wall": diffusing,
    "back_wall": diffusing,
    "ceiling": concrete,
}


########### trace the rays including scattering
simulator = ca.Simulator(scene, simulation_platform="pyroomacoustics")
rays = simulator.trace_rays(
    n_rays=100,
    max_reflection_order=4,
    include_scattering=True,
    scatter_density=12,
    scatter_max_bounces=1,
    random_seed=42,
)
print(f"Traced {len(rays)} specular rays and {len(rays.scattered_rays)} scattered rays.")


########### keep a single ray so the scattering is easy to see
single = rays.filter(source=source_1, ray_index=44)


########### display both the specular ray and its scattered rays, in distinct styles
single.display_in_viewer(
    scene=scene,
    show_specular=True,
    show_scattered=True,
    reflection_order=3,
)
