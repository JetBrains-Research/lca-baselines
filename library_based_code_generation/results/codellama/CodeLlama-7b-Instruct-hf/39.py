  import BurnMan

# Create a list of minerals and their molar abundances
minerals = [
    ("SiO2", 0.6),
    ("Al2O3", 0.3),
    ("FeO", 0.1),
    ("MgO", 0.05),
    ("CaO", 0.05)
]

# Create a composite mineral using simple mole fractions
composite1 = BurnMan.CompositeMineral(minerals, [0.6, 0.3, 0.1, 0.05, 0.05])

# Create a composite mineral using preset solutions
composite2 = BurnMan.CompositeMineral(minerals, [0.6, 0.3, 0.1, 0.05, 0.05], solution="sand")

# Define a custom solution
custom_solution = [
    ("SiO2", 0.6),
    ("Al2O3", 0.3),
    ("FeO", 0.1),
    ("MgO", 0.05),
    ("CaO", 0.05)
]
composite3 = BurnMan.CompositeMineral(minerals, custom_solution)

# Compute seismic velocities and other properties
burnman = BurnMan.BurnMan(minerals)
burnman.compute_seismic_velocities()
burnman.compute_density()
burnman.compute_geotherm()

# Compare to a seismic reference model
reference_model = BurnMan.SeismicModel("sand")
misfit = burnman.compare_to_model(reference_model)

# Plot the results
burnman.plot_results(pressure=True, vs=True, vphi=True, density=True, geotherm=True)
burnman.save_plot("results.png")

# Print the misfit
print(misfit)