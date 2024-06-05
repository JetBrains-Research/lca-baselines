```python
import logging
import numpy as np
import capytaine as cpt

# 1. Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 2. Create a mesh of a sphere and a floating body
radius = 1.0
center = (0, 0, 0)
resolution = (30, 30)
sphere_mesh = cpt.mesh_sphere(radius, center, resolution)
floating_body = cpt.FloatingBody(mesh=sphere_mesh, name="sphere")
floating_body.add_translation_dof(name='Heave')

# 3. Extract the immersed part of the mesh
immersed_body = floating_body.keep_immersed_part()

# 4. Set up a BEMSolver
solver = cpt.BEMSolver()

# 5. Define and solve a diffraction problem
wave_direction = 0.0
omega = 1.0
diffraction_problem = cpt.DiffractionProblem(body=immersed_body, wave_direction=wave_direction, omega=omega)
diffraction_result = solver.solve(diffraction_problem)

# 6. Define and solve a radiation problem
radiating_dof = 'Heave'
radiation_problem = cpt.RadiationProblem(body=immersed_body, radiating_dof=radiating_dof, omega=omega)
radiation_result = solver.solve(radiation_problem)

# 7. Define a free surface
x_range = (-5, 5)
y_range = (-5, 5)
n_x, n_y = 50, 50
free_surface = cpt.FreeSurface(x_range=x_range, y_range=y_range, n_x=n_x, n_y=n_y)

# 8. Compute the free surface elevation
diffraction_elevation = free_surface.elevation(diffraction_result)
radiation_elevation = free_surface.elevation(radiation_result)

# 9. Add incoming waves to the diffraction elevation
diffraction_elevation += free_surface.incoming_wave_elevation(diffraction_result)

# 10. Create and run animations
camera_position = (0, -10, 5)
for result, elevation, title in zip([diffraction_result, radiation_result], [diffraction_elevation, radiation_elevation], ['Diffraction', 'Radiation']):
    animation = cpt.plot.Animation(result, free_surface=free_surface, elevations=elevation, face_motion=result.body.linear_faces_motion_at_time(), camera_position=camera_position)
    animation.run(title=title)
```