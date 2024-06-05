 ```python
import burnman
import matplotlib.pyplot as plt

# Define reference seismic model
reference_model = burnman.models.PREM()

# Define minerals
olivine = burnman.minerals.olivine()
pyroxene = burnman.minerals.enstatite()
garnet = burnman.minerals.garnet()

# Example 1: Mix minerals in simple mole fractions
mole_fractions_1 = [0.7, 0.3]
mineral_1 = burnman.composite.Mixture([(olivine, mole_fractions_1[0]), (pyroxene, mole_fractions_1[1])])

# Example 2: Mix three minerals
mole_fractions_2 = [0.5, 0.3, 0.2]
mineral_2 = burnman.composite.Mixture([(olivine, mole_fractions_2[0]), (pyroxene, mole_fractions_2[1]), (garnet, mole_fractions_2[2])])

# Example 3: Use preset solutions
mineral_3 = burnman.composite.Solution(burnman.solutions.mgsi_perovskite())

# Example 4: Define a custom solution
custom_solution = burnman.composite.Solution(
    [
        ("MgSiO3_perovskite", 0.5),
        ("MgO_perovskite", 0.5),
    ],
    phase_list=["perovskite"],
)
mineral_4 = burnman.composite.Mixture([(custom_solution, 1.0)])

# Compute seismic velocities and other properties
composition = [(mineral_1, 0.5), (mineral_2, 0.3), (mineral_3, 0.1), (mineral_4, 0.1)]
properties = burnman.composite.Composite(composition)
seismic_model = burnman.SeismicModel(properties, [0.0, 140.0])

pressures = seismic_model.pressures
vs = seismic_model.vs
vphi = seismic_model.v_phi
density = seismic_model.density
geotherm = seismic_model.geotherm

# Calculate misfit between computed and reference values
misfit_vs = sum((vs - reference_model.vs)**2)
misfit_vphi = sum((vphi - reference_model.v_phi)**2)
misfit_density = sum((density - reference_model.density)**2)
misfit_geotherm = sum((geotherm - reference_model.geotherm)**2)

print(f"Misfit for Vs: {misfit_vs}")
print(f"Misfit for V_phi: {misfit_vphi}")
print(f"Misfit for density: {misfit_density}")
print(f"Misfit for geotherm: {misfit_geotherm}")

# Plot the computed and reference values
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(pressures, vs, label="Computed")
axs[0, 0].plot(reference_model.pressures, reference_model.vs, label="Reference")
axs[0, 0].set_xlabel("Pressure [GPa]")
axs[0, 0].set_ylabel("Vs [km/s]")
axs[0, 0].legend()

axs[0, 1].plot(pressures, vphi, label="Computed")
axs[0, 1].plot(reference_model.pressures, reference_model.v_phi, label="Reference")
axs[0, 1].set_xlabel("Pressure [GPa]")
axs[0, 1].set_ylabel("V_phi [km/s]")
axs[0, 1].legend()

axs[1, 0].plot(pressures, density, label="Computed")
axs[1, 0].plot(reference_model.pressures, reference_model.density, label="Reference")
axs[1, 0].set_xlabel("Pressure [GPa]")
axs[1, 0].set_ylabel("Density [g/cm^3]")
axs[1, 0].legend()

axs[1, 1].plot(pressures, geotherm, label="Computed")
axs[1, 1].plot(reference_model.pressures, reference_model.geotherm, label="Reference")
axs[1, 1].set_xlabel("Pressure [GPa]")
axs[1, 1].set_ylabel("Geotherm [K]")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig("seismic_model.png")
```