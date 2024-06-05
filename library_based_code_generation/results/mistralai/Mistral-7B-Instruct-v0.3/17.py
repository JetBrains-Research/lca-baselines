 ```python
from bluemira.core.api import *
from bluemira.electromagnetics.api import *
from fenics import *

# Define parameters
coil_radius = 0.05
coil_height = 0.02
enclosure_radius = 0.1
enclosure_height = 0.1
turns = 100
current = 1.0

# Create coil and enclosure geometries
coil_geometry = Cylinder(Pnt3D(0, 0, 0), coil_radius, coil_height)
enclosure_geometry = Cylinder(Pnt3D(0, 0, 0), enclosure_radius, enclosure_height)

# Set mesh options
coil_mesh_options = MeshOptions2D(type='tetrahedral', min_size=0.001, max_size=0.01)
enclosure_mesh_options = MeshOptions2D(type='tetrahedral', min_size=0.001, max_size=0.01)

# Create components
universe = Universe()
coil = universe.add_component('coil', coil_geometry, coil_mesh_options)
enclosure = universe.add_component('enclosure', enclosure_geometry, enclosure_mesh_options)

# Create mesh and convert for FEniCS
coil_mesh = coil.create_mesh()
enclosure_mesh = enclosure.create_mesh()
coil_mesh_fenics, enclosure_mesh_fenics = convert_mesh(coil_mesh, enclosure_mesh)

# Instantiate magnetostatic solver and define source term
solver = MagnetostaticSolver(coil_mesh_fenics, enclosure_mesh_fenics)
source_term = current * turns * DiracDelta((0, 0))

# Set up and solve the problem
solver.setup()
solver.solve(source_term)

# Calculate magnetic field
H = solver.get_magnetic_field()

# Theoretical magnetic field for a single coil
r = sqrt(x[0]**2 + x[1]**2)
z = x[2]
theoretical_H = (current * turns * M_PI) / (2 * r) * sin(M_PI * z / coil_height)

# Calculate and plot differences along z-axis and radial path
z_values = linspace(0, coil_height, 100)
r_values = linspace(0, coil_radius, 100)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(z_values, abs(H[2](Point(z_values, 0, 0)) - theoretical_H(z_values)), label='Z-axis')
axs[0].set_title('Difference along Z-axis')
axs[1].plot(r_values, abs(H[2](Point(0, r_values, coil_height/2)) - theoretical_H(r_values)), label='Radial path')
axs[1].set_title('Difference along Radial path')
axs[1].set_xlabel('Radius')
plt.legend()
plt.show()
```