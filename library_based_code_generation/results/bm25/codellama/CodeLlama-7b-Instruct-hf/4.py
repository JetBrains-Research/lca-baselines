  Here is the Python code that performs the tasks specified in the instructions using the capytaine library:
```
import capytaine as ct

# Set up logging
ct.set_logging_level(ct.LoggingLevel.INFO)
ct.set_logging_format("[%(levelname)s] %(message)s")

# Create a mesh of a sphere with a specified radius, center, and resolution
radius = 1.0
center = (0.0, 0.0, 0.0)
resolution = (10, 10, 10)
mesh = ct.mesh_sphere(radius, center, resolution)

# Create a floating body from the mesh and add a translation degree of freedom
body = ct.FloatingBody(mesh)
body.add_dof(ct.DegreeOfFreedom.TRANSLATION)

# Extract the immersed part of the mesh
immersed_part = body.get_immersed_part()

# Set up a BEMSolver
solver = ct.BEMSolver(immersed_part)

# Define and solve a diffraction problem for the immersed part of the sphere
wave_direction = (1.0, 0.0, 0.0)
omega = 1.0
diffraction_problem = ct.DiffractionProblem(immersed_part, wave_direction, omega)
diffraction_result = solver.solve(diffraction_problem)

# Define and solve a radiation problem for the immersed part of the sphere
radiating_degree_of_freedom = ct.DegreeOfFreedom.TRANSLATION
radiation_problem = ct.RadiationProblem(immersed_part, radiating_degree_of_freedom, omega)
radiation_result = solver.solve(radiation_problem)

# Define a free surface with specified x and y ranges and number of points in each direction
x_range = (-1.0, 1.0)
y_range = (-1.0, 1.0)
num_points = (100, 100)
free_surface = ct.FreeSurface(x_range, y_range, num_points)

# Compute the free surface elevation for both the diffraction and radiation results
diffraction_elevation = free_surface.compute_elevation(diffraction_result)
radiation_elevation = free_surface.compute_elevation(radiation_result)

# Add incoming waves to the diffraction elevation
incoming_waves = ct.IncomingWaves(diffraction_result)
diffraction_elevation += incoming_waves.get_elevation()

# Create and run two animations: one for the diffraction result and one for the radiation result
diffraction_animation = ct.Animation(diffraction_elevation, "Diffraction Result")
diffraction_animation.set_camera_position((0.0, 0.0, 1.0))
diffraction_animation.run()

radiation_animation = ct.Animation(radiation_elevation, "Radiation Result")
radiation_animation.set_camera_position((0.0, 0.0, 1.0))
radiation_animation.run()
```
Note that this code assumes that the capytaine library is installed and imported as `ct`.