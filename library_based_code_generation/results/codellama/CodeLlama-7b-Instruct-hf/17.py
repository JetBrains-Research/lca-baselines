  import bluemira as bm
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the coil and enclosure
R = 0.5  # radius of the coil
L = 1.0  # length of the coil
N = 100  # number of turns
A = np.pi * R**2  # cross-sectional area of the coil

# Create the coil and enclosure using bluemira's geometry tools
coil = bm.Coil(R, L, N)
enclosure = bm.Enclosure(coil, R)

# Set mesh options for each
coil.set_mesh_options(num_elements=100, element_size=0.01)
enclosure.set_mesh_options(num_elements=100, element_size=0.01)

# Create components for the universe, enclosure, and coil
universe = bm.Universe()
universe.add_component(coil)
universe.add_component(enclosure)

# Create a mesh and convert it for use in FEniCS
mesh = universe.create_mesh()
mesh = bm.convert_mesh(mesh, 'fenics')

# Instantiate a magnetostatic solver and define the source term for the problem
solver = bm.MagnetostaticSolver(mesh)
source = bm.Source(bm.SourceType.MAGNETIC, [0, 0, 1], [0, 0, 0])

# Solve the magnetostatic problem and calculate the magnetic field
solver.solve(source)
field = solver.get_field()

# Plot the magnetic field
plt.plot(field.x, field.y, field.z)
plt.xlabel('x')
plt.ylabel('y')
plt.zlabel('z')
plt.title('Magnetic Field')
plt.show()

# Compare the calculated magnetic field with the theoretical value
# along the z-axis and along a radial path at a certain z-offset
z_offset = 0.5
radial_path = np.linspace(0, 1, 100)
z_values = np.linspace(0, z_offset, 100)

# Calculate the theoretical magnetic field along the z-axis
theoretical_field_z = np.zeros(100)
for i in range(100):
    theoretical_field_z[i] = A / (2 * np.pi * z_values[i])

# Calculate the theoretical magnetic field along a radial path
theoretical_field_radial = np.zeros(100)
for i in range(100):
    theoretical_field_radial[i] = A / (2 * np.pi * radial_path[i])

# Plot the differences between the calculated and theoretical values
plt.plot(z_values, field.z - theoretical_field_z, label='z-axis')
plt.plot(radial_path, field.r - theoretical_field_radial, label='radial path')
plt.xlabel('z or radial path')
plt.ylabel('difference between calculated and theoretical values')
plt.legend()
plt.show()