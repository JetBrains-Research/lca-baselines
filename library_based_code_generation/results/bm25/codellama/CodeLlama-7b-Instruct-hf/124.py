  ```
import os
from pyaedt import Hfss

# Create a temporary folder and print its path
temp_folder = os.path.join(os.getenv("TEMP"), "AEDT_temp")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
print(temp_folder)

# Download an example file into the temporary folder
example_file = "example.aedt"
download_url = "https://www.example.com/example.aedt"
check_and_download_file(download_url, example_file, temp_folder)

# Set the non-graphical mode and launch AEDT in graphical mode using SI units
non_graphical = False
aedt_app = Hfss(example_file, non_graphical, new_desktop_session=True,
                close_on_exit=False, student_version=True,
                student_version_license=True,
                si_units=True)

# Initialize AEDT and launch HFSS 3D Layout
aedt_app.modeler.create_box(10, 10, 10, "MyBox")

# If the AEDT file already exists, remove it and save the project in the temporary folder
if os.path.exists(example_file):
    os.remove(example_file)
    aedt_app.save_project(example_file)

# Print the boundaries from the setups object
print(aedt_app.modeler.boundaries)

# Hide all nets and then make only two specified nets visible
aedt_app.modeler.hide_all_nets()
aedt_app.modeler.show_nets(["Net1", "Net2"])

# Plot the two specified nets
aedt_app.plot(show=True)

# Make all layers visible
aedt_app.modeler.layers.show_all_layers()

# Change the color of a specified layer
aedt_app.modeler.layers.set_layer_color("MyLayer", (255, 0, 0))

# Disable the visibility of components for the top and bottom layers
aedt_app.modeler.layers.set_layer_visibility("Top", False)
aedt_app.modeler.layers.set_layer_visibility("Bottom", False)

# Fit all so that all can be visualized
aedt_app.fit_all()

# Close the project and release the desktop
aedt_app.close_project(example_file)
aedt_app.release_desktop()
```