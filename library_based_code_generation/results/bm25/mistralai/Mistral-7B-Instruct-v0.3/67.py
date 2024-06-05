 Here is a Python script that uses the pyaedt library to create a 2D Extractor CPWG (coplanar waveguide with ground) design and runs a simulation.

```python
from pyaedt import *

# Set non-graphical mode
Aedt.set_option('non_graphical', True)

# Launch AEDT and 2D Extractor
app = CreateAedtApplication()
aedt = app.launch()

# Define variables
width = 10e-6
gap = 5e-6
length = 50e-6
substrate_thickness = 1e-6
substrate_permittivity = 10.2
conformal_thickness = 1e-6

# Create primitives
point_1 = point_a([0, 0])
point_2 = point_a([length, 0])
point_3 = point_a([length, width/2])
point_4 = point_a([0, width/2])

# Create coplanar ground
cpwg = create_waveguide(point_1, point_2, point_3, point_4, width, gap)

# Create reference ground plane
ref_ground = create_plane(point_1, point_2, point_3, point_4)
ref_ground.set_property('layer', 'GND')

# Create dielectric
dielectric = create_block(point_1, point_2, point_3, point_4, substrate_thickness, substrate_permittivity)

# Create conformal coating
conformal = _create_conformal(cpwg, conformal_thickness)

# Assign a conductor to the signal
signal = signal(cpwg)

# Create a reference ground
ref_ground_obj = CreateAedtBlockObj(ref_ground)
ref_ground_block = aedt.workspace.add_block(ref_ground_obj)

# Create Huray model on the signal
signal.set_property('material', 'Huray')

# Create the setup
setup = create_setup()

# Analyze it
setup.run_frequency_sweep(start_freq=1e9, end_freq=6e9, freq_points=100)

# Plot solution data
create_3d_plot(setup)

# Save the project
project = create_new_project('CPWG_Simulation')
project.save()

# Close AEDT
aedt.quit()
```

Please note that you need to have the pyaedt library installed and the AEDT software running to execute this script. Also, the Huray model might not be available in all versions of AEDT, so you might need to adjust the material property accordingly.

This script creates a 2D coplanar waveguide with ground (CPWG) design, sets up a frequency sweep simulation, and plots the 3D solution data. The project is then saved and AEDT is closed.