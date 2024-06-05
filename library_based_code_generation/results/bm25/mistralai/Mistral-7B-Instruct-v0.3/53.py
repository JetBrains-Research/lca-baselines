 Here is a basic example of how you might structure your code using the simpa library. Please note that this is a simplified version and you may need to adjust it according to your specific needs.

```python
from simpa.tissue_library import TissueLibrary
from simpa.tissue_properties import OpticalTissueProperties, MorphologicalTissueProperties
from simpa.simulation_module import SimulationModule
from simpa.linear_unmixing import LinearUnmixing
from simpa.test_functions import test_reconstruction_of_simulation, TestLinearUnmixing, TestLinearUnmixingVisual

# Global parameters
volume_dimensions = (10, 10, 10)
spacing = 1
wavelengths = [550, 600, 650]

# Create properties for each component
background_properties = OpticalTissueProperties(mu_a=0.01, mu_s_tot=0.02)
muscle_properties = OpticalTissueProperties(mu_a=0.02, mu_s_tot=0.05)
epidermis_properties = OpticalTissueProperties(mu_a=0.03, mu_s_tot=0.08)
blood_vessel_properties = OpticalTissueProperties(mu_a=0.1, mu_s_tot=0.2)

# Create morphological properties for each component
background_morphology = MorphologicalTissueProperties(shape='box', size=volume_dimensions)
muscle_morphology = MorphologicalTissueProperties(shape='box', size=(9, 9, 9), position=(1, 1, 1))
epidermis_morphology = MorphologicalTissueProperties(shape='box', size=(1, 1, 10), position=(0, 0, 0))
blood_vessel_morphology_1 = MorphologicalTissueProperties(shape='box', size=(1, 1, 1), position=(5, 5, 5))
blood_vessel_morphology_2 = MorphologicalTissueProperties(shape='box', size=(1, 1, 1), position=(5, 5, 9))

# Create tissue structure
tissue_structure = TissueLibrary.create_test_structure_of_molecule(
    background=background_properties,
    muscle=muscle_properties,
    epidermis=epidermis_properties,
    blood_vessels=[blood_vessel_properties, blood_vessel_properties]
)
tissue_structure.add_morphology(background_morphology)
tissue_structure.add_morphology(muscle_morphology)
tissue_structure.add_morphology(epidermis_morphology)
tissue_structure.add_morphology(blood_vessel_morphology_1)
tissue_structure.add_morphology(blood_vessel_morphology_2)

# Create simulation module
simulation_module = SimulationModule(tissue_structure, wavelengths=wavelengths, spacing=spacing)

# Create linear unmixing
linear_unmixing = LinearUnmixing(n_components=3)

# Run simulation and linear unmixing
simulation_results = simulation_module.run_simulation(linear_unmixing=linear_unmixing)

# Test simulation and linear unmixing
test_reconstruction_of_simulation(simulation_results, wavelengths)
TestLinearUnmixing(simulation_results, wavelengths)
TestLinearUnmixingVisual(simulation_results, wavelengths)

# Visualize results and linear unmixing
# You may need to use additional libraries like matplotlib or mayavi for visualization
```

This code creates a tissue structure with a background, a muscle layer, an epidermis layer, and two blood vessels. It then runs a simulation for the specified wavelengths and performs linear unmixing. The results of the simulation and the linear unmixing are tested, and you can visualize the results using appropriate libraries.