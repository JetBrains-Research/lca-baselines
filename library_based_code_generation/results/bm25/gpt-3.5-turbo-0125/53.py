import simpa

# Set up global parameters
volume_dimensions = (100, 100, 100)  # (x, y, z)
spacing = (1, 1, 1)  # (dx, dy, dz)
wavelengths = [500, 600, 700]  # in nm

# Function to create tissue structure
def create_tissue_structure():
    tissue = simpa.TissueLibrary()
    background = simpa.TissueProperties(background_absorption_coefficient=0.1, background_scattering_coefficient=10)
    muscle = simpa.TissueProperties(absorption_coefficient=0.5, scattering_coefficient=50)
    epidermis = simpa.TissueProperties(absorption_coefficient=0.2, scattering_coefficient=20)
    blood_vessel1 = simpa.TissueProperties(absorption_coefficient=0.8, scattering_coefficient=80)
    blood_vessel2 = simpa.TissueProperties(absorption_coefficient=0.8, scattering_coefficient=80)
    
    tissue.add_layer(background)
    tissue.add_layer(muscle)
    tissue.add_layer(epidermis)
    tissue.add_layer(blood_vessel1)
    tissue.add_layer(blood_vessel2)
    
    return tissue

# Create simulation volume
simulation_volume = simpa.create_simulation_volume(volume_dimensions, spacing)

# Run simulation for all wavelengths
simulation_results = simpa.run_simulation(simulation_volume, wavelengths, create_tissue_structure())

# Linear unmixing
linear_unmixing = simpa.LinearUnmixing()
linear_unmixing_results = linear_unmixing.run_linear_unmixing(simulation_results)

# Load and visualize results
simpa.TestLinearUnmixingVisual(linear_unmixing_results).visualize()