  ```
from dream import *

# Set up the electric field strength
E_field_strength = 1e6

# Set up the electron density and temperature
electron_density = 1e20
temperature = 1e6

# Define the momentum grid
momentum_grid = np.linspace(0, 10, 100)

# Set up the initial hot electron Maxwellian
hot_electron_maxwellian = Maxwellian(electron_density, temperature, momentum_grid)

# Include the Dreicer and avalanche terms in the equation system
equation_system = hot_electron_maxwellian + Dreicer(E_field_strength) + Avalanche(E_field_strength)

# Set up the radial grid and disable the runaway grid
radial_grid = np.linspace(0, 10, 100)
runaway_grid = None

# Set the Svensson transport coefficients and use the nonlinear solver
transport_coefficients = Svensson(electron_density, temperature, momentum_grid)
solver = NonlinearSolver(equation_system, transport_coefficients, radial_grid, runaway_grid)

# Set the time stepper and save the settings to an HDF5 file
time_stepper = TimeStepper(solver, 1e-3, 1e-3, 1e-3)
time_stepper.save_settings_to_hdf5("settings.h5")
```