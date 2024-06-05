import bluemira as bm

# Define parameters for the coil and enclosure
coil_radius = 1.0
coil_height = 1.0
enclosure_radius = 2.0
enclosure_height = 2.0

# Create coil and enclosure using bluemira's geometry tools
coil = bm.Cylinder(center=(0, 0, 0), radius=coil_radius, height=coil_height)
enclosure = bm.Cylinder(center=(0, 0, 0), radius=enclosure_radius, height=enclosure_height)

# Set mesh options for coil and enclosure
coil_mesh_options = bm.MeshOptions()
enclosure_mesh_options = bm.MeshOptions()

# Create components for universe, enclosure, and coil
universe = bm.Component()
universe.add(enclosure)
universe.add(coil)

# Create mesh and convert for FEniCS library
mesh = bm.Mesh()
mesh.convert_to_fenics()

# Instantiate magnetostatic solver
solver = bm.MagnetostaticSolver(mesh)

# Define source term for the problem
source_term = bm.SourceTerm()
source_term.plot()

# Solve magnetostatic problem and calculate magnetic field
solver.solve()
magnetic_field = solver.calculate_magnetic_field()

# Compare calculated magnetic field with theoretical value
z_axis_difference = solver.compare_with_theoretical_value(axis='z')
radial_path_difference = solver.compare_with_theoretical_value(axis='radial', z_offset=0.5)

# Plot the differences for each comparison
z_axis_difference.plot()
radial_path_difference.plot()