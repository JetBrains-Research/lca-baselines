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
example_url = "https://www.example.com/example.aedt"
if not os.path.exists(os.path.join(temp_folder, example_file)):
    urllib.request.urlretrieve(example_url, os.path.join(temp_folder, example_file))

# Set the non-graphical mode and launch AEDT in graphical mode using SI units
non_graphical = False
aedt_app = Hfss(non_graphical, new_desktop_session=True, project_path=temp_folder, project_name="example",
                design_name="example", solution_type="SteadyTemperatureAndFlow", units_override="SI")

# Initialize AEDT and launch HFSS 3D Layout
aedt_app.modeler.model_units = "mm"
aedt_app.modeler.model_units_override = "SI"
aedt_app.modeler.create_box(center=[0, 0, 0], size=[10, 10, 10], name="box")

# If the AEDT file already exists, remove it and save the project in the temporary folder
if os.path.exists(os.path.join(temp_folder, "example.aedt")):
    os.remove(os.path.join(temp_folder, "example.aedt"))
    aedt_app.save_project()

# Print the boundaries from the setups object
print(aedt_app.setups["Setup1"].boundaries)

# Hide all nets and then make only two specified nets visible
aedt_app.modeler.hide_all_nets()
aedt_app.modeler.set_net_visibility(["Net1", "Net2"], True)

# Plot the two specified nets
aedt_app.plot(show=False, setup_name="Setup1", plot_air_objects=False, plot_all_nets=False,
              plot_nets=["Net1", "Net2"], plot_deleted_objects=False, plot_reference_planes=False)

# Make all layers visible
aedt_app.modeler.set_all_layers_visibility(True)

# Change the color of a specified layer
aedt_app.modeler.set_layer_color("Layer1", [0, 0, 255])

# Disable the visibility of components for the top and bottom layers
aedt_app.modeler.set_component_visibility(["Top", "Bottom"], False)

# Fit all so that all can be visualized
aedt_app.fit_all()

# Close the project and release the desktop
aedt_app.close_project(save_project=False)
aedt_app.release_desktop()
```