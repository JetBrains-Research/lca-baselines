  import BurnMan

# Create a new BurnMan instance
bm = BurnMan.BurnMan()

# Define the minerals and their molar abundances
minerals = ['SiO2', 'Al2O3', 'FeO', 'MgO', 'CaO']
abundances = [0.5, 0.3, 0.1, 0.05, 0.05]

# Create a composite mineral using simple mole fractions
composite_mineral = bm.polynomial_esv_and_mbr_mixing(minerals, abundances)

# Create a composite mineral using preset solutions
composite_mineral_2 = bm.instantiate_minerals(['SiO2', 'Al2O3', 'FeO', 'MgO', 'CaO'], [0.5, 0.3, 0.1, 0.05, 0.05])

# Define a custom solution
custom_solution = ['SiO2', 'Al2O3', 'FeO', 'MgO', 'CaO']
custom_abundances = [0.5, 0.3, 0.1, 0.05, 0.05]
composite_mineral_3 = bm.set_compositions_and_state_from_parameters(custom_solution, custom_abundances)

# Compute seismic velocities and other properties
bm.set_minerals(composite_mineral)
bm.compute_seismic_properties()

# Compare the computed and reference values
bm.compare_to_reference_model()

# Plot the computed and reference values of Vs, Vphi, density, and geotherm against pressure
bm.plot_results()

# Save the plot as an image file
bm.save_plot('plot.png')

# Compute the misfit between the computed and reference values
bm.compute_misfit()

# Print the misfit
print(bm.misfit)