from burnman import Mineral, SeismicTable, test_seismic
import matplotlib.pyplot as plt

# Create different minerals
mineral1 = Mineral.from_formula('MgSiO3', 'bridgmanite')
mineral2 = Mineral.from_formula('MgO', 'periclase')
mineral3 = Mineral.from_formula('FeO', 'wustite')

# Compute seismic velocities
minerals = [mineral1, mineral2, mineral3]
abundances = [0.5, 0.3, 0.2]
vs, vphi, density, geotherm = test_seismic(minerals, abundances)

# Compare to seismic reference model
seismic_table = SeismicTable()
reference_vs = seismic_table.lookup_and_interpolate('Vs', geotherm)
reference_vphi = seismic_table.lookup_and_interpolate('Vphi', geotherm)
reference_density = seismic_table.lookup_and_interpolate('density', geotherm)

# Calculate misfit
misfit_vs = vs - reference_vs
misfit_vphi = vphi - reference_vphi
misfit_density = density - reference_density

# Plot computed and reference values
plt.plot(geotherm, vs, label='Computed Vs')
plt.plot(geotherm, reference_vs, label='Reference Vs')
plt.plot(geotherm, vphi, label='Computed Vphi')
plt.plot(geotherm, reference_vphi, label='Reference Vphi')
plt.plot(geotherm, density, label='Computed Density')
plt.plot(geotherm, reference_density, label='Reference Density')
plt.xlabel('Pressure')
plt.ylabel('Values')
plt.legend()
plt.savefig('seismic_properties.png')
plt.show()