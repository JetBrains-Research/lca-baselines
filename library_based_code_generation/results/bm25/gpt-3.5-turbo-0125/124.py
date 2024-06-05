import os
from pyaedt import Desktop

temp_folder = os.path.join(os.getcwd(), "temp_folder")
print("Temporary folder path:", temp_folder)

example_file_url = "http://example.com/example_file.aedt"
temp_file_path = os.path.join(temp_folder, "example_file.aedt")

# Download example file into temporary folder
Desktop.check_and_download_file(example_file_url, temp_file_path)

# Launch AEDT in graphical mode with SI units
d = Desktop("2022.1", non_graphical=False, specified_version="2022.1", specified_units="si")

# Initialize AEDT and launch HFSS 3D Layout
hfss = d.Hfss3dLayout()

# Remove existing AEDT file and save project in temporary folder
if os.path.exists(temp_file_path):
    os.remove(temp_file_path)
hfss.save_project(temp_file_path)

# Print boundaries from setups object
print(hfss.setups.boundaries)

# Hide all nets and make only two specified nets visible
hfss.nets.hide_all_nets()
hfss.nets.set_visibility(["Net1", "Net2"], visibility=True)

# Plot the two specified nets
hfss.nets.plot(["Net1", "Net2"])

# Make all layers visible
hfss.all_layers.set_all_layers_visibility(visibility=True)

# Change color of a specified layer
hfss.all_layers.change_layer_color("Layer1", (255, 0, 0))

# Disable visibility of components for top and bottom layers
hfss.all_layers.disable_layer_components(["TopLayer", "BottomLayer"])

# Fit all to visualize
hfss.fit_all()

# Close project and release desktop
hfss.close_project()
d.release_desktop()