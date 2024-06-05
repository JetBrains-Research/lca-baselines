 Here is the Python code that performs the tasks as per your instructions using the capytaine library:

```python
import capytaine as cp
import numpy as np

# Set up logging
cp.set_logger(level='INFO', format='%(asctime)s - %(message)s')

# Define parameters
radius = 1.0
center = [0, 0, 0]
resolution = 64
omega = 2 * np.pi
k = omega / cp.C0
wave_direction = [np.cos(k * 30), np.sin(k * 30), 0]

# Create a sphere mesh and floating body
mesh = cp.mesh_sphere(radius, center, resolution)
body = cp.FloatingBody(mesh, translation_dofs=True)

# Extract the immersed part of the mesh
immersed_mesh = body.immersed_part()

# Set up BEMSolver
solver = cp.BEMSolver()

# Define and solve diffraction and radiation problems
diffraction_problem = cp.DiffractionProblem(immersed_mesh, wave_direction, omega)
diffraction_result = solver.solve(diffraction_problem)

radiation_problem = cp.RadiationProblem(immersed_mesh, omega)
radiation_result = solver.solve(radiation_problem)

# Define free surface
free_surface = cp.FreeSurface(x_range=[-radius, radius], y_range=[-radius, radius], nx=128, ny=128)

# Compute free surface elevation for both diffraction and radiation results
diffraction_elevation = cp.compute_free_surface_elevation(diffraction_result, free_surface)
radiation_elevation = cp.compute_free_surface_elevation(radiation_result, free_surface)

# Add incoming waves to the diffraction elevation
diffraction_elevation += cp.incoming_waves(free_surface, wave_direction, omega)

# Create and run animations
cp.animate(diffraction_result, free_surface, camera_position=[10, 10, 10], face_motions=['rotation_x', 'rotation_y'], elevations=diffraction_elevation)
cp.animate(radiation_result, free_surface, camera_position=[10, 10, 10], face_motions=['rotation_x', 'rotation_y'], elevations=radiation_elevation)
```

Please note that you need to have the capytaine library installed to run this code. If you haven't installed it yet, you can do so using pip:

```
pip install capytaine
```