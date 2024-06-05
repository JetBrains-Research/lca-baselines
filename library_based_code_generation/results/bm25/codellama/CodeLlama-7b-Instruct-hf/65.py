  ```
from dream import *

# Set up simulation parameters
T_initial = 1e-6
T_final = 1e-4
dt = 1e-12
n_re = 1e19
ion_species = 'p'
E_field = 1e-3
cold_electron_temperature = 1e-3

# Set up radial grid
radial_grid = RadialGrid(n_points=100, r_min=1e-6, r_max=1e-4)

# Set up time stepper
time_stepper = construct_A(radial_grid, dt)

# Set up ions
ions = loadCODE(ion_species)

# Set up E_field
E_field = E_field * np.ones(len(radial_grid))

# Set up cold electron temperature
cold_electron_temperature = cold_electron_temperature * np.ones(len(radial_grid))

# Set up transport settings
transport_settings = TransportSettings(n_re=n_re,
                                       diffusion_coefficient=1e-3,
                                       ion_species=ion_species,
                                       E_field=E_field,
                                       cold_electron_temperature=cold_electron_temperature)

# Set up hot electron distribution
hot_electron_distribution = HotElectronDistribution(radial_grid,
                                                    ion_species,
                                                    E_field,
                                                    cold_electron_temperature)

# Set up hot tail grid
hot_tail_grid = setHot(radial_grid, hot_electron_distribution)

# Set up transport coefficients
transport_coefficients = TransportCoefficientReader(radial_grid,
                                                    ion_species,
                                                    E_field,
                                                    cold_electron_temperature)

# Set up simulation
simulation = runSimulation(radial_grid,
                           time_stepper,
                           ions,
                           transport_settings,
                           hot_tail_grid,
                           transport_coefficients)

# Run simulation
simulation.run(T_initial, T_final)

# Plot results
plot(simulation.get_temperature(T_initial, T_final))
```