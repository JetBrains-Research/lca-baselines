import numpy as np
import dream
import dream.simulation
import dream.initialcondition
import dream.parameters
import dream.output
import dream.radialgrid
import dream.transport
import dream.transport.coefficients
import dream.time_stepping
import dream.hot_tail_grid

# Set up simulation parameters
n_re_initial = 1e19
n_re_final = 1e20
time_steps = 1000
ion_species = ['D']
E_field = 1.0
cold_electron_temperature = 1.0

# Set up radial grid
nr = 100
rgrid = dream.radialgrid.RadialGrid(nr=nr)

# Set up time stepper
time_stepper = dream.time_stepping.TimeStepper()

# Set up ions
ions = dream.parameters.Ions(ion_species)

# Set E_field and cold electron temperature
E_field = dream.parameters.E_field(E_field)
cold_electron_temperature = dream.parameters.ColdElectronTemperature(cold_electron_temperature)

# Enable hot tail grid
hot_tail_grid = dream.hot_tail_grid.HotTailGrid()

# Set up transport settings
transport_settings = dream.transport.TransportSettings()

# Run simulation
simulation = dream.simulation.Simulation(
    initial_condition=dream.initialcondition.ColdBeam(
        n_re_initial=n_re_initial,
        n_re_final=n_re_final,
        rgrid=rgrid
    ),
    time_stepper=time_stepper,
    ions=ions,
    E_field=E_field,
    cold_electron_temperature=cold_electron_temperature,
    hot_tail_grid=hot_tail_grid,
    transport_settings=transport_settings
)

# Conditions for different transport modes
if transport_settings.transport_model == dream.transport.TransportModel.CONSTANT:
    print("Using constant transport model")
else:
    print("Using other transport model")

# Check if hot tail grid is enabled
if hot_tail_grid.enabled:
    print("Hot tail grid is enabled")
else:
    print("Hot tail grid is not enabled")

# Run simulation
simulation.run(time_steps)