 ```python
import numpy as np
import matplotlib.pyplot as plt
from burnman import Composite, Mineral, SeismicModel

# Define minerals
cp = Mineral('cpx', 'cpx_diopside', 0.9, 300e3, 298.15)
opx = Mineral('opx', 'enstatite', 0.9, 300e3, 298.15)
grt = Mineral('grt', 'garnet_pyrope', 0.9, 300e3, 298.15)
sp = Mineral('sp', 'stishovite', 0.9, 300e3, 298.15)

# Define composite minerals
composite1 = Composite.from_mole_fractions([cp, opx], [0.7, 0.3])
composite2 = Composite.from_mole_fractions([cp, opx, grt], [0.6, 0.3, 0.1])
composite3 = Composite.from_solution('garnet', 'garnet_pyrope_almandine_spessartine', 0.9, 300e3, 298.15)
composite4 = Composite()
composite4.add_mineral(sp, 0.5)
composite4.add_mineral(cp, 0.5)
composite4.set_state('upper_mantle')

# Define seismic model
reference_model = SeismicModel('ak135')

# Compute seismic velocities and other properties
composite_states = [composite1, composite2, composite3, composite4]
pressures = np.linspace(0, 140e3, 100)
vs, vp, density, geotherm = [], [], [], []
for state in composite_states:
    state.set_state('upper_mantle')
    vs.append([state.evaluate('Vs', p) for p in pressures])
    vp.append([state.evaluate('Vp', p) for p in pressures])
    density.append([state.evaluate('density', p) for p in pressures])
    geotherm.append([state.evaluate('geotherm', p) for p in pressures])

# Calculate misfit
misfit = np.mean((np.array(vs) - reference_model.vs(pressures))**2)

# Plot the computed and reference values
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(pressures, np.array(vs).T, label='Vs')
axs[0, 0].plot(pressures, reference_model.vs(pressures), 'k--', label='Reference')
axs[0, 0].set_ylabel('Vs (km/s)')
axs[0, 0].legend()

axs[0, 1].plot(pressures, np.array(vp).T, label='Vp')
axs[0, 1].plot(pressures, reference_model.vp(pressures), 'k--', label='Reference')
axs[0, 1].set_ylabel('Vp (km/s)')
axs[0, 1].legend()

axs[1, 0].plot(pressures, np.array(density).T, label='Density')
axs[1, 0].plot(pressures, reference_model.rho(pressures), 'k--', label='Reference')
axs[1, 0].set_ylabel('Density (kg/m3)')
axs[1, 0].legend()

axs[1, 1].plot(pressures, np.array(geotherm).T, label='Geotherm')
axs[1, 1].plot(pressures, reference_model.T(pressures), 'k--', label='Reference')
axs[1, 1].set_ylabel('Temperature (K)')
axs[1, 1].legend()
plt.tight_layout()
plt.savefig('seismic_properties.png')

print(f'Misfit between computed and reference Vs: {misfit:.2e}')
```