  import numpy as np
import matplotlib.pyplot as plt
from dream import *

# Set up the simulation
sim = Simulation()

# Set up the radial grid
sim.grid.radial = np.linspace(0, 10, 100)

# Set up the time steps
sim.time.dt = 0.01
sim.time.tmax = 100

# Add ions
sim.ions.add(1, 1, 1)

# Set up the temperature and electric field
sim.temperature.set_profile(0, 100, 1000)
sim.electric.set_profile(0, 100, 1000)

# Disable runaway and hot-tail grid
sim.grid.runaway = False
sim.grid.hot_tail = False

# Initialize the simulation
sim.initialize()

# Ionize the plasma
sim.ions.ionize(1, 1, 1)

# Equilibrate the plasma
sim.equilibrate()

# Calculate the ohmic heating and radiative losses
sim.ohmic_heating.calculate()
sim.radiative_losses.calculate()

# Plot the results
plt.plot(sim.temperature.profile, sim.ohmic_heating.profile, label='Ohmic Heating')
plt.plot(sim.temperature.profile, sim.radiative_losses.profile, label='Radiative Losses')
plt.xlabel('Temperature (eV)')
plt.ylabel('Heating/Losses (eV)')
plt.legend()
plt.show()