  import capytaine

# Set up logging
capytaine.logging.set_level(capytaine.logging.Level.INFO)
capytaine.logging.set_format("[%(levelname)s] %(message)s")

# Create a mesh of a sphere
radius = 1.0
center = (0.0, 0.0, 0.0)
resolution = (10, 10, 10)
mesh = capytaine.mesh_sphere(radius, center, resolution)

# Create a floating body from the mesh
body = capytaine.FloatingBody(mesh)

# Add a translation degree of freedom to the body
body.add_dof(capytaine.DegreeOfFreedom.TRANSLATION)

# Extract the immersed part of the mesh
immersed_part = body.get_immersed_part()

# Set up a BEMSolver
solver = capytaine.BEMSolver(immersed_part)

# Define and solve a diffraction problem
wave_direction = (1.0, 0.0, 0.0)
omega = 1.0
solver.add_diffraction_problem(wave_direction, omega)
solver.solve()
diffraction_result = solver.get_diffraction_result()

# Define and solve a radiation problem
radiating_degree_of_freedom = capytaine.DegreeOfFreedom.ROTATION
solver.add_radiation_problem(radiating_degree_of_freedom, omega)
solver.solve()
radiation_result = solver.get_radiation_result()

# Define a free surface
x_range = (-1.0, 1.0)
y_range = (-1.0, 1.0)
num_points = (100, 100)
free_surface = capytaine.FreeSurface(x_range, y_range, num_points)

# Compute the free surface elevation for both the diffraction and radiation results
diffraction_elevation = free_surface.compute_elevation(diffraction_result)
radiation_elevation = free_surface.compute_elevation(radiation_result)

# Add incoming waves to the diffraction elevation
incoming_waves = capytaine.IncomingWaves(diffraction_result)
diffraction_elevation.add_incoming_waves(incoming_waves)

# Create and run two animations
animation_1 = capytaine.Animation(diffraction_result, free_surface, body)
animation_1.set_camera_position((0.0, 0.0, 1.0))
animation_1.run()

animation_2 = capytaine.Animation(radiation_result, free_surface, body)
animation_2.set_camera_position((0.0, 0.0, 1.0))
animation_2.run()