import bluemira

# Define parameters for the coil and enclosure
coil_radius = 1.0
coil_height = 2.0
enclosure_radius = 5.0
enclosure_height = 10.0

# Create coil and enclosure using geometry tools
coil = bluemira.Coil(radius=coil_radius, height=coil_height)
enclosure = bluemira.Enclosure(radius=enclosure_radius, height=enclosure_height)

# Set mesh options for coil and enclosure
coil.set_mesh_options(mesh_size=0.1)
enclosure.set_mesh_options(mesh_size=0.2)

# Create components for universe, enclosure, and coil
universe = bluemira.Universe()
universe.add_component(enclosure)
universe.add_component(coil)

# Create mesh and convert for FEniCS library
mesh = bluemira.create_mesh(universe)
fenics_mesh = bluemira.convert_mesh_for_fenics(mesh)

# Instantiate magnetostatic solver
solver = bluemira.MagnetostaticSolver(mesh=fenics_mesh)

# Define source term for the problem
source_term = bluemira.define_source_term()
bluemira.plot_source_term(source_term)

# Solve magnetostatic problem and calculate magnetic field
solver.solve()
magnetic_field = solver.calculate_magnetic_field()

# Compare calculated magnetic field with theoretical value
z_axis_difference = bluemira.compare_along_z_axis(magnetic_field)
radial_path_difference = bluemira.compare_along_radial_path(magnetic_field, z_offset=5.0)

# Plot differences for each comparison
bluemira.plot_differences(z_axis_difference)
bluemira.plot_differences(radial_path_difference)