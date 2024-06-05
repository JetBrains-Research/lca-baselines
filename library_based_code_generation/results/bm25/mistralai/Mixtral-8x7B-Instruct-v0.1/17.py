 ```python
import bluemira
import bluemira.geometry.primitives as primitives
import bluemira.magnetostatics as magstat
import matplotlib.pyplot as plt
import numpy as np

# Define parameters
coil_radius = 0.05
coil_turns = 50
enclosure_radius = 0.15
gap_size = 0.02
mesh_size = 0.01

# Create coil and enclosure
coil = primitives.coil(radii=(coil_radius,), turns=coil_turns)
enclosure = primitives.circle(radius=enclosure_radius) - primitives.circle(radius=coil_radius + gap_size)

# Set mesh options
coil_mesh_opts = bluemira.geometry.mesh.MeshOptions(max_elem_edge_length=mesh_size)
enclosure_mesh_opts = bluemira.geometry.mesh.MeshOptions(max_elem_edge_length=mesh_size)

# Create components
universe = bluemira.structures.Universe(primitives=[coil, enclosure])
enclosure_comp = bluemira.structures.Component(name="enclosure", primitives=[enclosure])
coil_comp = bluemira.structures.Component(name="coil", primitives=[coil])

# Create mesh and convert for FEniCS
bl_mesh = bluemira.geometry.mesh.mesh_from_components([universe], mesh_options=[coil_mesh_opts, enclosure_mesh_opts])
fenics_mesh = bluemira.convert.bl_mesh_to_fenics(bl_mesh)

# Instantiate magnetostatic solver
solver = magstat.Magnetostatics(fenics_mesh)

# Define source term
source_term = magstat.CurrentDensity(coil_comp, value=1 / (2 * np.pi * coil_radius * coil_turns))

# Plot source term for visualization
fig, ax = plt.subplots()
source_term.plot_volume(axes=ax)
plt.show()

# Solve magnetostatic problem and calculate magnetic field
solver.solve(source_term=source_term)
B_calc = solver.get_magnetic_field()

# Calculate theoretical magnetic field
B_theory_z = 0.5 * coil_turns * coil_radius / (gap_size ** 2 + coil_radius ** 2)
B_theory_radial = B_theory_z * (gap_size / coil_radius)

# Compare calculated and theoretical magnetic field along z-axis
z_vals = np.linspace(-gap_size, gap_size, 100)
B_calc_z = [B_calc.norm(z_vals[i], 0, 0) for i in range(len(z_vals))]
B_theory_z_vals = [B_theory_z for _ in z_vals]
diff_z = np.abs(np.array(B_calc_z) - np.array(B_theory_z_vals))

# Compare calculated and theoretical magnetic field along radial path at z-offset
z_offset = 0.01
radial_vals = np.linspace(0, coil_radius + gap_size, 100)
B_calc_radial = [B_calc.norm(z_offset, radial_vals[i], 0) for i in range(len(radial_vals))]
B_theory_radial_vals = [B_theory_radial for _ in radial_vals]
diff_radial = np.abs(np.array(B_calc_radial) - np.array(B_theory_radial_vals))

# Plot differences
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(z_vals, diff_z)
ax1.set_xlabel("z-axis position")
ax1.set_ylabel("Difference in magnetic field")
ax1.set_title("Difference along z-axis")

ax2.plot(radial_vals, diff_radial)
ax2.set_xlabel("Radial position")
ax2.set_ylabel("Difference in magnetic field")
ax2.set_title("Difference along radial path at z-offset")
plt.tight_layout()
plt.show()
```