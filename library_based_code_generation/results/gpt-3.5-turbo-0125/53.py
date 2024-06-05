import simpa

def create_tissue_structure():
    # Create background
    background = simpa.Background()
    
    # Create muscle layer
    muscle = simpa.MuscleLayer()
    
    # Create epidermis layer
    epidermis = simpa.EpidermisLayer()
    
    # Create blood vessels
    blood_vessel1 = simpa.BloodVessel()
    blood_vessel2 = simpa.BloodVessel()
    
    # Add components to tissue structure
    tissue_structure = simpa.TissueStructure()
    tissue_structure.add_layer(background)
    tissue_structure.add_layer(muscle)
    tissue_structure.add_layer(epidermis)
    tissue_structure.add_layer(blood_vessel1)
    tissue_structure.add_layer(blood_vessel2)
    
    return tissue_structure

# Set global parameters
volume_dimensions = (100, 100, 100)  # in mm
spacing = 0.1  # in mm
wavelengths = [650, 700, 750]  # in nm

# Create tissue structure
tissue_structure = create_tissue_structure()

# Run simulation for all wavelengths
results = simpa.run_simulation(tissue_structure, volume_dimensions, spacing, wavelengths)

# Perform linear unmixing
unmixed_results = simpa.linear_unmixing(results)

# Load and visualize results
simpa.load_results(results)
simpa.load_results(unmixed_results)
simpa.visualize_results()