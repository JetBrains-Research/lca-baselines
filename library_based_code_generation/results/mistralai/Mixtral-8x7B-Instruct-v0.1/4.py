 ```python
import capytaine
import logging
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 1. Set up logging with a specific level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
radius = 10
center = np.array([0, 0, 0])
resolution = 32
omega = 2 * np.pi
wave_direction = np.array([1, 0, 0])
radiating_dof = 0
face_motion_diffraction = 0.1
face_motion_radiation = 0.05
camera_position = np.array([30, 30, 30])
x_range = [-20, 20]
y_range = [-20, 20]
x_points = 400
y_points = 400

# 2. Create a mesh of a sphere with a specified radius, center, and resolution
sphere_mesh = capytaine.mesh_sphere(radius, center, resolution)

# Create a floating body from this mesh and add a translation degree of freedom to it
body = capytaine.FloatingBody(sphere_mesh)
body.add_translation_dof(0)

# 3. Extract the immersed part of the mesh
immersed_mesh = body.immersed_part()

# 4. Set up a BEMSolver
solver = capytaine.BEMSolver()

# 5. Define and solve a diffraction problem for the immersed part of the sphere
diffraction_problem = capytaine.DiffractionProblem(immersed_mesh, omega, wave_direction)
solver.solve(diffraction_problem)

# 6. Define and solve a radiation problem for the immersed part of the sphere
radiation_problem = capytaine.RadiationProblem(immersed_mesh, omega, radiating_dof)
solver.solve(radiation_problem)

# 7. Define a free surface with specified x and y ranges and number of points in each direction
free_surface = capytaine.FreeSurface(x_range, y_range, x_points, y_points)

# 8. Compute the free surface elevation for both the diffraction and radiation results
diffraction_elevation = free_surface.elevation_from_potential(diffraction_problem.potentials)
radiation_elevation = free_surface.elevation_from_potential(radiation_problem.potentials)

# 9. Add incoming waves to the diffraction elevation
diffraction_elevation += capytaine.incoming_wave(omega, wave_direction, free_surface)

# 10. Create and run two animations: one for the diffraction result and one for the radiation result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update_plot(data, title):
    ax.clear()
    ax.plot_surface(data[0], data[1], data[2], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.set_title(title)

def animate_diffraction(i):
    update_plot(diffraction_elevation[:, :, i], 'Diffraction')

def animate_radiation(i):
    update_plot(radiation_elevation[:, :, i], 'Radiation')

capytaine.animate(animate_diffraction, animate_radiation, face_motions=[face_motion_diffraction, face_motion_radiation],
                  camera_positions=[camera_position, camera_position], interval=50)

plt.show()
```