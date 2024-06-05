 ```python
import numpy as np
import matplotlib.pyplot as plt

import bluemira as bm
from bluemira.geometry.primitives import Circle, Rectangle
from bluemira.mesh.create import create_mesh
from bluemira.mesh.refine import refine_mesh
from bluemira.structmech.solvers import StaticSolver
from bluemira.topology.optimize import optimize_topology

# Define parameters
coil_radius = 0.05
enclosure_radius = 0.1
enclosure_height = 0.02
n_turns = 50
current = 1

# Create coil and enclosure
coil = Circle(center=(0, 0), radius=coil_radius)
enclosure = Rectangle(
    lower_left=(-enclosure_radius, -enclosure_height / 2),
    upper_right=(enclosure_radius, enclosure_height / 2),
)

# Set mesh options
coil_mesh_opts = {
    "type": "tri",
    "size": 0.005,
    "grading": 1.2,
    "regions": {"coil": coil},
}
enclosure_mesh_opts = {
    "type": "tri",
    "size": 0.01,
    "grading": 1.1,
    "regions": {"enclosure": enclosure},
}

# Create components and universe
universe = bm.Region(name="universe")
enclosure_comp = bm.Region(name="enclosure", geometry=enclosure)
coil_comp = bm.Region(
    name="coil", geometry=coil, n_elements=n_turns, circumferential_element="line"
)
universe.add_region([enclosure_comp, coil_comp])

# Create mesh and convert for FEniCS
mesh = create_mesh(universe, **coil_mesh_opts)
mesh = refine_mesh(mesh, factor=2)
bm.convert(mesh, "dolfinx")

# Instantiate magnetostatic solver
solver = StaticSolver(
    problem=bm.Problem(
        geometry=universe,
        region_properties={"coil": {"current": current}},
        formulation="magnetostatics",
    )
)

# Define source term and plot
source_term = solver.problem.region_properties["coil"]["current"] * bm.dof_coordinate("x")
bm.plot_field(source_term, "source_term")

# Solve magnetostatic problem
solver.solve()

# Calculate magnetic field
Bz = bm.Field(name="Bz", dof_coordinate="z", region=universe)
solver.problem.project_expression(Bz, bm.magnetic_field("z"))

# Compare calculated magnetic field with theoretical value
theoretical_Bz = (mu_0 * current * n_turns) / (2 * np.pi * coil_radius)
z_values = np.linspace(-0.1, 0.1, 100)
radial_path = np.sqrt(enclosure_radius**2 - z_values**2)
z_offset = 0.05

diff_z = theoretical_Bz * np.ones_like(z_values) - Bz.evaluate(x=z_values, y=0)
diff_radial = theoretical_Bz * np.ones_like(z_values) - Bz.evaluate(
    x=radial_path, y=z_values + z_offset
)

# Plot differences
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(z_values, diff_z)
axs[0].set_xlabel("z-axis")
axs[0].set_ylabel("Bz difference")
axs[0].set_title("Difference along z-axis")

axs[1].plot(z_values, diff_radial)
axs[1].set_xlabel("z-axis")
axs[1].set_ylabel("Bz difference")
axs[1].set_title("Difference along radial path")
plt.tight_layout()
plt.show()
```