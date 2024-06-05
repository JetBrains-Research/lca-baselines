```python
import matplotlib.pyplot as plt
from burnman import minerals, mineralsdb, geotherm, mineral_helpers

# Create different minerals
mineral1 = mineralsdb.get_phase('MgSiO3', 'bridgmanite')
mineral2 = mineralsdb.get_phase('MgSiO3', 'perovskite')
mineral3 = mineralsdb.get_phase('Fe2O3', 'hematite')

# Compute seismic velocities
minerals_list = [mineral1, mineral2]
molar_abundances = [0.5, 0.5]
vs, vphi, density = mineral_helpers.compute_seismic_properties(minerals_list, molar_abundances)

# Compare to seismic reference model
reference_model = geotherm.ReferenceModel()
misfit = mineral_helpers.compute_misfit(vs, vphi, density, reference_model)

# Plot computed and reference values
pressure_range = [0, 100]
vs_values = [vs(p) for p in pressure_range]
vphi_values = [vphi(p) for p in pressure_range]
density_values = [density(p) for p in pressure_range]
geotherm_values = [reference_model.temperature(p) for p in pressure_range]

plt.plot(pressure_range, vs_values, label='Vs')
plt.plot(pressure_range, vphi_values, label='Vphi')
plt.plot(pressure_range, density_values, label='Density')
plt.plot(pressure_range, geotherm_values, label='Geotherm')
plt.legend()
plt.savefig('seismic_properties.png')
```