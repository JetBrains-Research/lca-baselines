import numpy as np
import matplotlib.pyplot as plt
import dream
import dream.simulation
import dream.parameters
import dream.initialcondition
import dream.radialgrid
import dream.time
import dream.output
import dream.transport
import dream.hottailgrid

# Set up simulation parameters
n_re = 1e19
initial_temperature = 1e3
final_temperature = 1e3
time_steps = 1000
ion_species = ['D']
E_field = 1.0
cold_electron_temperature = 1e3

# Set up radial grid
nr = 256
radial_grid = dream.radialgrid.RadialGrid(nr=nr)

# Set up time stepper
time_stepper = dream.time.TimeStepper(time_steps)

# Set up ions
ions = dream.parameters.Ions(ion_species)

# Set E_field and cold electron temperature
E_field = np.ones(nr) * E_field
cold_electrons = dream.initialcondition.ColdElectrons(cold_electron_temperature)

# Enable hot tail grid
hot_tail_grid = dream.hottailgrid.HotTailGrid(radial_grid)

# Set up transport settings
transport = dream.transport.TransportRadialDiffusion(1.0)

# Run simulation
simulation = dream.simulation.Simulation(radial_grid, time_stepper, ions, E_field, cold_electrons, hot_tail_grid, transport)
simulation.run()

# Conditions for different transport modes
if transport.mode == 'radial_diffusion':
    print('Radial diffusion transport mode enabled')
else:
    print('Unknown transport mode')

# Check if hot tail grid is enabled
if hot_tail_grid.enabled:
    print('Hot tail grid enabled')
else:
    print('Hot tail grid not enabled')