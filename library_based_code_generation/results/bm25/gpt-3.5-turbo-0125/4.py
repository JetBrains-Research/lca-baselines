import logging
from capytaine import *
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create mesh of a sphere
radius = 1.0
center = [0.0, 0.0, 0.0]
resolution = 32
mesh = mesh_sphere(radius=radius, center=center, resolution=resolution)
body = FloatingBody(mesh)
body.add_translation_dof()

# Extract immersed part of the mesh
immersed_mesh = body.mesh.immersed_part()

# Set up BEMSolver
solver = BEMSolver()

# Define and solve diffraction problem
wave_direction = np.array([1.0, 0.0, 0.0])
omega = 1.0
diffraction_problem = DiffractionProblem(immersed_part=immersed_mesh, wave_direction=wave_direction, omega=omega)
diffraction_result = solver.solve(diffraction_problem)

# Define and solve radiation problem
radiating_dof = 0
radiation_problem = RadiationProblem(immersed_part=immersed_mesh, radiating_dof=radiating_dof, omega=omega)
radiation_result = solver.solve(radiation_problem)

# Define free surface
x_range = [-2.0, 2.0]
y_range = [-2.0, 2.0]
num_points = 100
free_surface = FreeSurface(x_range=x_range, y_range=y_range, num_points=num_points)

# Compute free surface elevation
diffraction_elevation = diffraction_result.compute_free_surface_elevation(free_surface)
radiation_elevation = radiation_result.compute_free_surface_elevation(free_surface)

# Add incoming waves to diffraction elevation
incoming_waves = np.zeros(num_points)
diffraction_elevation_with_waves = diffraction_elevation + incoming_waves

# Create and run animations
diffraction_animation = Animation(diffraction_result, body, free_surface, face_motions=None, elevation=diffraction_elevation, camera_position=None)
diffraction_animation.run()

radiation_animation = Animation(radiation_result, body, free_surface, face_motions=None, elevation=radiation_elevation, camera_position=None)
radiation_animation.run()