```python
import numpy as np
from simpa import *

# Define global settings
global_settings = Settings()
global_settings[Tags.RANDOM_SEED] = 123
global_settings[Tags.VOLUME_NAME] = "TissueSimulation"
global_settings[Tags.SIMULATION_PATH] = "simulations"
global_settings[Tags.SPACING_MM] = 0.1
global_settings[Tags.DIM_VOLUME_Z_MM] = 30
global_settings[Tags.DIM_VOLUME_X_MM] = 30
global_settings[Tags.DIM_VOLUME_Y_MM] = 30
global_settings[Tags.WAVELENGTHS] = [700, 800, 900]

def create_tissue_structure():
    tissue_dict = {
        Tags.BACKGROUND: TISSUE_LIBRARY.constant(0.01, 0.9, 0.9, 0.9),
        Tags.MUSCLE: TISSUE_LIBRARY.muscle(),
        Tags.EPIDERMIS: TISSUE_LIBRARY.epidermis(),
        Tags.STRUCTURES: [
            Circle(settings=Settings({
                Tags.PRIORITY: 4,
                Tags.STRUCTURE_START_MM: [15, 15, 5],
                Tags.STRUCTURE_RADIUS_MM: 5,
                Tags.CIRCLE_BORDER_MM: 2,
                Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.blood(oxygenation=85)
            })),
            Circle(settings=Settings({
                Tags.PRIORITY: 4,
                Tags.STRUCTURE_START_MM: [10, 10, 10],
                Tags.STRUCTURE_RADIUS_MM: 3,
                Tags.CIRCLE_BORDER_MM: 1,
                Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.blood(oxygenation=70)
            }))
        ]
    }
    return tissue_dict

# Create the simulation pipeline
pipeline = [
    AbsorptionAndScatteringCreator(global_settings),
    GaussianNoise(settings=Settings({Tags.NOISE_MEAN: 0, Tags.NOISE_STD: 0.01})),
    OpticalForwardModel(settings=Settings({Tags.SOLVER: Tags.OPTICAL_MODEL_MCML})),
    LinearUnmixing(settings=Settings({Tags.WAVELENGTHS: global_settings[Tags.WAVELENGTHS]}))
]

# Run the simulation
simulate(global_settings, pipeline, create_tissue_structure())

# Load and visualize the results
volume_path = build_hdf5_file_path(global_settings[Tags.VOLUME_NAME], global_settings[Tags.SIMULATION_PATH])
volume = load_data_field(volume_path, Tags.DATA_FIELD_ABSORPTION_PER_CM)

import matplotlib.pyplot as plt

for wavelength in global_settings[Tags.WAVELENGTHS]:
    plt.figure()
    plt.title(f"Absorption at {wavelength}nm")
    plt.imshow(volume[:, :, volume.shape[2]//2, global_settings[Tags.WAVELENGTHS].index(wavelength)], cmap='hot')
    plt.colorbar()
    plt.show()
```