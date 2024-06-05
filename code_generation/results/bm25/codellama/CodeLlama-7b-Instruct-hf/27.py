  import dream

# Initialize the simulation
sim = dream.Simulation()

# Set up the radial grid
sim.radialGrid = dream.RadialGrid(100, 0.1, 10)

# Set up the time steps
sim.timeSteps = dream.TimeSteps(1e-4, 1e-4, 1e-4)

# Add ions
sim.addIons(dream.Ion('He', 1), dream.Ion('He', 2))

# Set up the temperature and electric field
sim.setTemperature(dream.Temperature(1e6, 'K'))
sim.setElectricField(dream.ElectricField(0, 0, 0))

# Disable runaway and hot-tail grid
sim.disableRunawayGrid()
sim.disableHotTailGrid()

# Initialize the simulation
sim.runSimulation()

# Ionize the plasma
sim.setIonization(dream.Ionization('Thermal', 1e6))
sim.runSimulation()

# Equilibrate the plasma
sim.setIonization(dream.Ionization('Thermal', 1e6))
sim.runSimulation()

# Radiate the plasma
sim.setIonization(dream.Ionization('Thermal', 1e6))
sim.runSimulation()

# Plot the energy balance
sim.plotEnergyBalance()

# Plot the temperature
sim.plotTemperature()

# Plot the electric field
sim.plotElectricField()

# Save the settings
sim.saveSettings('settings.json')

# Run the simulation again with different settings
sim.loadSettings('settings.json')
sim.setIonization(dream.Ionization('Thermal', 1e7))
sim.runSimulation()

# Plot the energy balance again
sim.plotEnergyBalance()

# Plot the temperature again
sim.plotTemperature()

# Plot the electric field again
sim.plotElectricField()