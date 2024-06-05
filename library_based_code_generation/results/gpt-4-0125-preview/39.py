```python
import numpy as np
import matplotlib.pyplot as plt
import burnman
from burnman import minerals

# Define pressure and temperature range
depths = np.linspace(0, 2890e3, 100)  # From the surface to the CMB
temperatures = burnman.geotherm.brown_shankland(depths)
pressures = burnman.seismology.PREM().evaluate(depths)['pressure']

# Example 1: Two minerals mixed in simple mole fractions
olivine = minerals.SLB_2011.mg_fe_perovskite()
periclase = minerals.SLB_2011.ferropericlase()
mixture1 = burnman.Composite([olivine, periclase], [0.6, 0.4])

# Example 2: Mix of three minerals
pyroxene = minerals.SLB_2011.enstatite()
garnet = minerals.SLB_2011.pyrope()
mixture2 = burnman.Composite([olivine, pyroxene, garnet], [0.5, 0.3, 0.2])

# Example 3: Using preset solutions
mixture3 = minerals.SLB_2011.mg_fe_perovskite()

# Example 4: Defining a custom solution
class CustomSolution(burnman.Solution):
    def __init__(self):
        endmembers = [[minerals.SLB_2011.forsterite(), 0.9],
                      [minerals.SLB_2011.fayalite(), 0.1]]
        burnman.Solution.__init__(self, endmembers)
custom_solution = CustomSolution()

# Compute seismic velocities and other properties
mixtures = [mixture1, mixture2, mixture3, custom_solution]
properties = ['v_p', 'v_s', 'density', 'v_phi']
results = {prop: [] for prop in properties}
for mixture in mixtures:
    mixture.set_state(pressures, temperatures)
    for prop in properties:
        results[prop].append(getattr(mixture, prop)())

# Compare to seismic reference model
prem = burnman.seismology.PREM()
reference_values = {'v_p': prem.evaluate(depths)['v_p'],
                    'v_s': prem.evaluate(depths)['v_s'],
                    'density': prem.evaluate(depths)['density']}

# Calculate misfit
misfits = {prop: [] for prop in properties[:-1]}  # v_phi not in reference
for prop in misfits:
    for result in results[prop]:
        misfits[prop].append(np.sqrt(np.mean((result - reference_values[prop])**2)))

# Plotting
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, prop in zip(axs, properties):
    for result in results[prop]:
        ax.plot(depths, result, label=prop)
    if prop in reference_values:
        ax.plot(depths, reference_values[prop], 'k--', label='PREM ' + prop)
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel(prop)
    ax.legend()

plt.tight_layout()
plt.savefig('seismic_properties_comparison.png')
```