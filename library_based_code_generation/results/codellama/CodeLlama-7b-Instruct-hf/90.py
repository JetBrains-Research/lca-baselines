  import numpy as np
import dream

# Set up simulation parameters
params = dream.SimulationParameters()
params.set_run_type('self-consistent')
params.set_time_step_size(1e-12)
params.set_max_timestep(1e-10)
params.set_max_duration(1e-9)

# Set up radial grid
grid = dream.RadialGrid()
grid.set_minor_radius(0.1)
grid.set_major_radius(10)
grid.set_minor_radius_delta(0.01)
grid.set_major_radius_delta(0.1)

# Set up time stepper
stepper = dream.TimeStepper()
stepper.set_type('rk4')
stepper.set_max_steps(1000)

# Add ions
ions = dream.IonHandler()
ions.add_ion('D', 1)

# Set electric field and cold electron temperature
efield = dream.ElectricField()
efield.set_type('gaussian')
efield.set_amplitude(1e-3)
efield.set_frequency(1e6)

cold_electron_temperature = dream.TemperatureProfile()
cold_electron_temperature.set_type('constant')
cold_electron_temperature.set_value(1e-3)

# Set up hot tail grid
hot_tail_grid = dream.HotTailGrid()
hot_tail_grid.set_type('none')

# Disable runaway grid
runaway_grid = dream.RunawayGrid()
runaway_grid.set_type('none')

# Set solver type and parameters
solver = dream.Solver()
solver.set_type('cg')
solver.set_max_iterations(1000)
solver.set_tolerance(1e-10)

# Set other necessary settings
settings = dream.Settings()
settings.set_grid(grid)
settings.set_stepper(stepper)
settings.set_ions(ions)
settings.set_electric_field(efield)
settings.set_cold_electron_temperature(cold_electron_temperature)
settings.set_hot_tail_grid(hot_tail_grid)
settings.set_runaway_grid(runaway_grid)
settings.set_solver(solver)

# Save settings to HDF5 file
settings.save('settings.h5')

# Run simulation
sim = dream.Simulation(settings)
sim.run()

# Restart simulation twice, adjusting time stepper and saving new settings
for i in range(2):
    # Load settings from previous run
    settings.load('settings.h5')

    # Adjust time stepper
    stepper.set_max_steps(stepper.get_max_steps() * 2)

    # Save new settings to HDF5 file
    settings.save('settings.h5')

    # Run simulation
    sim = dream.Simulation(settings)