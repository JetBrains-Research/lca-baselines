import logging
import capytaine as cpt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

radius = 1.0
center = [0.0, 0.0, 0.0]
resolution = 32

mesh = cpt.mesh_sphere(radius=radius, center=center, resolution=resolution)
body = cpt.FloatingBody(mesh)
body.add_translation_dof()

immersed_part = body.mesh.immersed_part()

solver = cpt.BEMSolver()

wave_direction = [1.0, 0.0, 0.0]
omega = 1.0
diffraction_problem = cpt.DiffractionProblem(immersed_part, wave_direction, omega)
diffraction_solution = solver.solve(diffraction_problem)

radiating_dof = 0
radiation_problem = cpt.RadiationProblem(immersed_part, radiating_dof, omega)
radiation_solution = solver.solve(radiation_problem)

x_range = [-2.0, 2.0]
y_range = [-2.0, 2.0]
n_points = 100
free_surface = cpt.FreeSurface(x_range, y_range, n_points)

diffraction_elevation = diffraction_solution.compute_free_surface_elevation(free_surface)
radiation_elevation = radiation_solution.compute_free_surface_elevation(free_surface)

incoming_waves = [0.1, 0.2, 0.3]
diffraction_elevation_with_waves = diffraction_elevation + incoming_waves

diffraction_animation = cpt.Animation(diffraction_solution, body, free_surface, face_motions=None, elevation=diffraction_elevation_with_waves, camera_position=[0.0, 0.0, 5.0])
diffraction_animation.run()

radiation_animation = cpt.Animation(radiation_solution, body, free_surface, face_motions=None, elevation=radiation_elevation, camera_position=[0.0, 0.0, 5.0])
radiation_animation.run()