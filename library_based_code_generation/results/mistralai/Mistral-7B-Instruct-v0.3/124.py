 Here is the Python code using the pyaedt library that performs the tasks as described:

```python
import os
import shutil
import pyaedt

# Create a temporary folder and print its path
temp_folder = os.path.join(os.getcwd(), 'temp_pyaedt')
os.makedirs(temp_folder, exist_ok=True)
print(temp_folder)

# Download an example file into the temporary folder
pyaedt.download_example_file(os.path.join(temp_folder, 'example.prj'))

# Set the non-graphical mode and launch AEDT in graphical mode using SI units
pyaedt.set_non_graphical_mode(False)
pyaedt.set_units('SI')
app = pyaedt.launch_application()

# Initialize AEDT and launch HFSS 3D Layout
project = pyaedt.Project(os.path.join(temp_folder, 'example.prj'))
project.open()
layout = project.get_layout()

# If the AEDT file already exists, it removes it and saves the project in the temporary folder
if os.path.exists(os.path.join(temp_folder, 'example.prj')):
    os.remove(os.path.join(temp_folder, 'example.prj'))
    project.save(os.path.join(temp_folder, 'example.prj'))

# Print the boundaries from the setups object
print(layout.setups.boundaries)

# Hide all nets and then make only two specified nets visible
layout.hide_nets()
layout.show_net('Net1')
layout.show_net('Net2')

# Plot the two specified nets
layout.plot_nets(['Net1', 'Net2'])

# Make all layers visible
layout.show_layers()

# Change the color of a specified layer
layer = layout.get_layer('Layer1')
layer.color = (1, 0, 0)

# Disable the visibility of components for the top and bottom layers
top_layer = layout.get_layer('Top Metal')
bottom_layer = layout.get_layer('Bottom Metal')
top_layer.visible = False
bottom_layer.visible = False

# Fit all so that all can be visualized
layout.fit_all()

# Close the project and release the desktop
project.close()
app.quit()
```

Please note that you need to replace 'Net1', 'Net2', and 'Layer1' with the actual net and layer names you want to work with. Also, make sure to install the pyaedt library before running the code.

You can install pyaedt using the following command:

```
pip install pyaedt
```