```python
import numpy as np
import matplotlib.pyplot as plt
from simpa import *

# Define global simulation settings
settings = Settings()
settings[Tags.VOLUME_NAME] = "TissueSimulation"
settings[Tags.SIMULATION_PATH] = "simulations"
settings[Tags.SPACING_MM] = 0.1
settings[Tags.DIM_VOLUME_X_MM] = 30
settings[Tags.DIM_VOLUME_Y_MM] = 30
settings[Tags.DIM_VOLUME_Z_MM] = 30
settings[Tags.WAVELENGTHS] = [700, 800, 900]

# Define the tissue structure
def create_tissue_structure():
    tissue_dict = {
        Tags.BACKGROUND: TissueProperties(name="background", optical_properties=OpticalTissueProperties(mua=0.01, mus=1, g=0.9, refractive_index=1.0)),
        "muscle": TissueProperties(name="muscle", optical_properties=OpticalTissueProperties(mua=0.02, mus=2, g=0.85, refractive_index=1.4)),
        "epidermis": TissueProperties(name="epidermis", optical_properties=OpticalTissueProperties(mua=0.03, mus=1.5, g=0.8, refractive_index=1.35)),
        "blood_vessel_1": TissueProperties(name="blood_vessel", optical_properties=OpticalTissueProperties(mua=0.15, mus=10, g=0.9, refractive_index=1.36)),
        "blood_vessel_2": TissueProperties(name="blood_vessel", optical_properties=OpticalTissueProperties(mua=0.15, mus=10, g=0.9, refractive_index=1.36))
    }
    return tissue_dict

# Create the simulation volume
def create_simulation_volume(tissue_properties):
    volume_creator = VolumeCreator(settings)
    volume = volume_creator.create_volume(tissue_properties)
    return volume

# Run the simulation
def run_simulation_for_all_wavelengths(settings, tissue_properties):
    simulation_results = []
    for wavelength in settings[Tags.WAVELENGTHS]:
        settings[Tags.WAVELENGTH] = wavelength
        simulation = MCXAdapter(settings)
        result = simulation.run(tissue_properties)
        simulation_results.append(result)
    return simulation_results

# Perform linear unmixing
def perform_linear_unmixing(simulation_results):
    unmixing_results = []
    for result in simulation_results:
        unmixer = LinearUnmixing()
        unmixed = unmixer.unmix(result)
        unmixing_results.append(unmixed)
    return unmixing_results

# Load and visualize the results
def visualize_results(simulation_results, unmixing_results):
    for i, (simulation, unmixing) in enumerate(zip(simulation_results, unmixing_results)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(simulation[:, :, int(simulation.shape[2]/2)], cmap='gray')
        plt.title(f'Simulation Result Wavelength {settings[Tags.WAVELENGTHS][i]}')
        plt.subplot(1, 2, 2)
        plt.imshow(unmixing[:, :, int(unmixing.shape[2]/2)], cmap='gray')
        plt.title(f'Unmixing Result Wavelength {settings[Tags.WAVELENGTHS][i]}')
        plt.show()

# Main function to run the simulation
def main():
    tissue_properties = create_tissue_structure()
    simulation_volume = create_simulation_volume(tissue_properties)
    simulation_results = run_simulation_for_all_wavelengths(settings, tissue_properties)
    unmixing_results = perform_linear_unmixing(simulation_results)
    visualize_results(simulation_results, unmixing_results)

if __name__ == "__main__":
    main()
```