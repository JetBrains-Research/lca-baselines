  import simpa

# Define global parameters for the simulation
volume_dimensions = (100, 100, 100)
spacing = (1, 1, 1)
wavelengths = [400, 500, 600]

# Define the tissue structure with specific properties for each component
def create_tissue_structure(tissue_library):
    tissue_library.add_background(0.1)
    tissue_library.add_muscle(0.5, 0.1)
    tissue_library.add_epidermis(0.3, 0.2)
    tissue_library.add_blood_vessels(0.2, 0.3)

# Create the simulation volume with the defined tissue structure
simulation_volume = simpa.create_simulation_volume(volume_dimensions, spacing, create_tissue_structure)

# Run the simulation for all wavelengths
for wavelength in wavelengths:
    simulation_volume.run_simulation(wavelength)

# Perform linear unmixing on the simulation results
linear_unmixing = simpa.LinearUnmixing(simulation_volume.get_simulation_results())
linear_unmixing.run_linear_unmixing()

# Load and visualize the simulation results and linear unmixing
simulation_results = linear_unmixing.get_linear_unmixing_results()
simulation_volume.visualize_simulation_results(simulation_results)

# Test the simulation and linear unmixing components
simpa.test_simulation(simulation_volume)
simpa.test_linear_unmixing(linear_unmixing)

# Test the tissue library methods
simpa.get_all_tissue_library_methods()

# Simulate and evaluate the tissue structure with a device
simpa.simulate_and_evaluate_with_device(simulation_volume, device_name='MyDevice')

# Create a test structure of molecule
simpa.create_test_structure_of_molecule(molecule_name='MyMolecule', structure_type='MyStructureType')

# Test the layer structure partial volume close to border
simpa.test_layer_structure_partial_volume_close_to_border(simulation_volume)

# Create a simple tissue model
simpa.create_simple_tissue_model(tissue_name='MyTissue', tissue_properties=simpa.TissueProperties())

# Test the write and read structure dictionary
simpa.test_write_and_read_structure_dictionary(simulation_volume)

# Test the optical tissue properties
simpa.test_optical_tissue_properties(simulation_volume)

# Test the morphological tissue properties
simpa.test_morphological_tissue_properties(simulation_volume)