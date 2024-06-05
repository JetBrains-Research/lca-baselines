 ```python
import pyaedt
import numpy as np

pyaedt.set_non_graphical_mode()

# Launch AEDT and set properties
session = pyaedt.Session(version='2020.1')
project = session.workspace.open_project('new_project.prj')

# Set properties
project.property_manager.set_property('General', 'Units', 'Millimeters')
project.property_manager.set_property('General', 'Model Units', 'Millimeters')
project.property_manager.set_property('General', 'Open Region', 'On')
project.property_manager.set_property('General', 'Automatically Use Causal Materials', 'On')
project.property_manager.set_property('General', 'Material Override', 'On')

# Define variables
total_length = 100
theta = np.pi / 4
radius = 1
width = 0.5
height = 0.1
spacing = 0.5
ground_width = 1
ground_thickness = 0.1

# Function to create bending
def create_bending(radius, extension):
    bending = project.geometry.create_bending(radius=radius, extension=extension)
    return bending

# Create signal line and ground lines
signal_line = project.geometry.create_coplanar_waveguide(total_length=total_length, width=width, height=height, spacing=spacing)
ground_line_1 = project.geometry.create_ground_plane(x=total_length / 2, y=0, width=ground_width, thickness=ground_thickness)
ground_line_2 = create_bending(radius, extension=total_length / 2)
ground_line_3 = create_bending(radius, extension=-total_length / 2)

# Create dielectric and bottom metals
dielectric = project.geometry.create_rectangle(x=0, y=0, width=total_length, height=height)
bottom_metal = project.geometry.create_ground_plane(x=0, y=height, width=total_length, thickness=ground_thickness)

# Create port interfaces and Perfect E boundary condition
port_1 = project.geometry.create_port(x=0, y=0, size=width, type='PEC')
port_2 = project.geometry.create_port(x=total_length, y=0, size=width, type='PEC')
perfect_e_boundary = project.geometry.create_perfect_electric_boundary(x=total_length + width, y=0, width=project.geometry.get_boundary_width())

# Create ports and setup and sweep
frequency_start = 1
frequency_stop = 10
frequency_step = 0.1
sweep = project.analysis.create_frequency_sweep(start=frequency_start, stop=frequency_stop, step=frequency_step)
port_1.set_port_type('Incident')
port_2.set_port_type('Scattered')

# Plot the model and release AEDT
project.plot_3d()
project.run_job()
```