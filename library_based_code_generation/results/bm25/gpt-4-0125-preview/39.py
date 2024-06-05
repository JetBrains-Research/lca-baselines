```python
import numpy as np
import matplotlib.pyplot as plt
import burnman
from burnman import minerals

# Define pressure and temperature range
pressures = np.linspace(1e9, 25e9, 100)  # in Pa
temperatures = 1900 + pressures * 3e-12  # an arbitrary geotherm

# Example 1: Two minerals mixed in simple mole fractions
olivine = minerals.SLB_2011.forsterite()
wadsleyite = minerals.SLB_2011.wadsleyite()
mole_fractions = [0.5, 0.5]
composite1 = burnman.Composite([olivine, wadsleyite], mole_fractions)

# Example 2: Mix of three minerals
ringwoodite = minerals.SLB_2011.ringwoodite()
mole_fractions_3 = [0.4, 0.4, 0.2]
composite2 = burnman.Composite([olivine, wadsleyite, ringwoodite], mole_fractions_3)

# Example 3: Using preset solutions
pyrolite = burnman.minerals.SLB_2011.mg_bridgmanite() + burnman.minerals.SLB_2011.fpericlase()

# Example 4: Defining a custom solution
class CustomSolution(burnman.Solution):
    def __init__(self):
        endmembers = [[minerals.SLB_2011.forsterite(), 0.5], [minerals.SLB_2011.fayalite(), 0.5]]
        burnman.Solution.__init__(self, endmembers)
custom_solution = CustomSolution()

# Compute seismic velocities and other properties
composites = [composite1, composite2, pyrolite, custom_solution]
labels = ['50% Forsterite - 50% Wadsleyite', '40% Forsterite - 40% Wadsleyite - 20% Ringwoodite', 'Pyrolite', 'Custom Solution']
colors = ['r', 'g', 'b', 'm']

for composite, label, color in zip(composites, labels, colors):
    composite.set_state(pressures, temperatures)
    vp, vs, rho = composite.evaluate(['v_p', 'v_s', 'density'])
    plt.plot(pressures, vs, label=label, color=color)

# Compare to a seismic reference model
prem = burnman.seismic.PREM()
depths = np.linspace(0, 2890e3, len(pressures))
reference_vp, reference_vs, reference_rho = prem.evaluate(['v_p', 'v_s', 'density'], depths)

plt.plot(pressures, reference_vs, label='PREM', linestyle='--', color='k')

# Plotting
plt.xlabel('Pressure (Pa)')
plt.ylabel('Shear Velocity (m/s)')
plt.title('Shear Velocity vs. Pressure')
plt.legend()
plt.savefig('seismic_velocities_vs_pressure.png')
plt.show()
```