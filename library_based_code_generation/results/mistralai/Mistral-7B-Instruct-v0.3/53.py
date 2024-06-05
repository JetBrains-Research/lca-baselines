 Here is a simplified example of how you might set up a simulation using the Simpa library. Please note that this is a basic example and might need adjustments based on your specific requirements. Also, I'm assuming you have Simpa installed and the necessary imports are in place.

```python
import simpa
import numpy as np

# Global parameters
volume_size = (100, 100, 100)  # Volume dimensions
spacing = 1  # Voxel spacing
wavelengths = np.array([600, 700, 800])  # Wavelengths for simulation

def create_tissue_structure():
    background = simpa.Material('background', absorptions=np.ones(3))
    muscle = simpa.Material('muscle', absorptions=np.ones(3))
    epidermis = simpa.Material('epidermis', absorptions=np.ones(3))
    blood_vessel1 = simpa.Material('blood_vessel1', absorptions=np.ones(3))
    blood_vessel2 = simpa.Material('blood_vessel2', absorptions=np.ones(3))

    # Define geometry
    geometry = simpa.Geometry()
    geometry.add_box('background', volume_size, material=background)
    geometry.add_box('muscle', (90, 90, 90), material=muscle, center=(50, 50, 50))
    geometry.add_box('epidermis', (92, 92, 92), material=epidermis, center=(51, 51, 51))
    geometry.add_cylinder('blood_vessel1', (20, 20), 100, material=blood_vessel1, center=(50, 50, 50))
    geometry.add_cylinder('blood_vessel2', (20, 20), 100, material=blood_vessel2, center=(50, 50, 95))

    return geometry

def run_simulation():
    geometry = create_tissue_structure()
    sim = simpa.Simulation(geometry, wavelengths=wavelengths)
    sim.run()
    return sim

def linear_unmixing(sim):
    results = sim.results()
    unmixed = simpa.linear_unmixing(results)
    return unmixed

def visualize_results(unmixed):
    # Visualization code here (e.g., matplotlib, mayavi, etc.)
    pass

# Run the simulation and visualize the results
sim = run_simulation()
unmixed = linear_unmixing(sim)
visualize_results(unmixed)
```

This code creates a tissue structure with a background, a muscle layer, an epidermis layer, and two blood vessels. It then runs the simulation for the specified wavelengths and performs linear unmixing. The results are not visualized in this example, as you would need to implement the visualization part according to your preferred library (e.g., matplotlib, mayavi, etc.).