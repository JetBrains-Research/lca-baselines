  import dream

# Set up simulation parameters
n_re = 1e19
T_i = 1e-6
T_f = 1e-3
dt = 1e-12
N_r = 1000
N_t = 10000

# Set up ion species
ion = dream.Ion(n_re, 'n_re')

# Set up electric field
E_field = dream.ElectricField(0, 0, 0)

# Set up cold electron temperature
T_cold = 1e-3

# Set up radial grid
r_grid = dream.RadialGrid(N_r, 1e-6, 1e-3)

# Set up time stepper
time_stepper = dream.TimeStepper(N_t, dt)

# Set up ions
ions = dream.IonHandler(ion, r_grid, time_stepper)

# Set up transport settings
transport_settings = dream.TransportSettings(T_i, T_f, n_re, r_grid, time_stepper)

# Set up simulation
simulation = dream.Simulation(transport_settings, ions, E_field, T_cold)

# Enable hot tail grid
simulation.enable_hot_tail_grid()

# Set up conditions for different transport modes
transport_modes = ['adiabatic', 'non-adiabatic']
for transport_mode in transport_modes:
    simulation.set_transport_mode(transport_mode)
    simulation.run()

# Disable hot tail grid
simulation.disable_hot_tail_grid()

# Set up conditions for different ion species
ion_species = ['n_re', 'p_re']
for ion_species in ion_species:
    simulation.set_ion_species(ion_species)
    simulation.run()

# Set up conditions for different electric fields
E_fields = [0, 1e-3, 2e-3]
for E_field in E_fields:
    simulation.set_E_field(E_field)
    simulation.run()

# Set up conditions for different cold electron temperatures
T_cold_values = [1e-3, 2e-3, 3e-3]
for T_cold in T_cold_values:
    simulation.set_T_cold(T_cold)
    simulation.run()