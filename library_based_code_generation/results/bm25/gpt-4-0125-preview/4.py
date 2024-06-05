```python
import logging
import numpy as np
from capytaine import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Create a mesh of a sphere
radius = 1.0
center = (0, 0, 0)
resolution = (30, 30)
sphere_mesh = mesh_sphere(radius=radius, center=center, ntheta=resolution[0], nphi=resolution[1])

# Create a floating body from this mesh and add a translation degree of freedom
sphere_body = FloatingBody(mesh=sphere_mesh, name='sphere')
sphere_body.add_translation_dof(name='Heave', axis=(0, 0, 1))

# Extract the immersed part of the mesh
immersed_sphere = sphere_body.keep_immersed_part()

# Set up a BEMSolver
solver = BEMSolver()

# Define and solve a diffraction problem
wave_direction = 0.0
omega = 2.0
diffraction_problem = DiffractionProblem(body=immersed_sphere, wave_direction=wave_direction, omega=omega)
diffraction_result = solver.solve(diffraction_problem)

# Define and solve a radiation problem
radiating_dof = 'Heave'
radiation_problem = RadiationProblem(body=immersed_sphere, radiating_dof=radiating_dof, omega=omega)
radiation_result = solver.solve(radiation_problem)

# Define a free surface
x_range = (-5, 5)
y_range = (-5, 5)
n_points = (50, 50)
free_surface = FreeSurface(x_range=x_range, y_range=y_range, n_points=n_points)

# Compute the free surface elevation for both the diffraction and radiation results
diffraction_elevation = diffraction_result.free_surface_elevation(free_surface)
radiation_elevation = radiation_result.free_surface_elevation(free_surface)

# Add incoming waves to the diffraction elevation
diffraction_elevation += diffraction_result.incoming_wave_elevation(free_surface)

# Create and run animations
camera_position = (0, -10, 5)
for result, elevation, title in [(diffraction_result, diffraction_elevation, 'Diffraction'),
                                 (radiation_result, radiation_elevation, 'Radiation')]:
    animation = result.plot_free_surface_elevation(
        sphere_body, free_surface, elevation=elevation, face_motion=result.body_motion_at_faces,
        camera_position=camera_position, title=title
    )
    animation.start()
```