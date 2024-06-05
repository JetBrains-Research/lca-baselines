 import capytaine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

radius = 1.0
center = [0.0, 0.0, 0.0]
resolution = 32
sphere_mesh = capytaine.mesh_sphere(radius, center, resolution)
floating_body = capytaine.FloatingBody.from_mesh(sphere_mesh, density=1.0)
floating_body.add_translation_dof(center)

immersed_sphere = floating_body.immersed_part()

solver = capytaine.BEMSolver()

wave_direction = [1.0, 0.0, 0.0]
omega = 1.0
diffraction_problem = capytaine.DiffractionProblem(immersed_sphere, wave_direction, omega)
diffraction_result = solver.solve(diffraction_problem)

radiating_dof = 0
radiation_problem = capytaine.RadiationProblem(immersed_sphere, radiating_dof, omega)
radiation_result = solver.solve(radiation_problem)

x_range = [-5.0, 5.0]
y_range = [-5.0, 5.0]
npoints = 100
free_surface = capytaine.FreeSurface.from_rectangle(x_range, y_range, npoints, npoints)

diffraction_elevation = free_surface.compute_elevation(diffraction_result)
radiation_elevation = free_surface.compute_elevation(radiation_result)

incoming_wave = capytaine.IncomingWaves.from_direction(wave_direction, omega)
diffraction_elevation += incoming_wave.elevation

diffraction_animation = capytaine.Animation(
    immersed_geometry=immersed_sphere,
    free_surface=free_surface,
    face_motion=diffraction_result.displacements,
    elevation=diffraction_elevation,
    camera_position=[5.0, 5.0, 5.0]
)

radiation_animation = capytaine.Animation(
    immersed_geometry=immersed_sphere,
    free_surface=free_surface,
    face_motion=radiation_result.displacements,
    elevation=radiation_elevation,
    camera_position=[5.0, 5.0, 5.0]
)

diffraction_animation.run()
radiation_animation.run()