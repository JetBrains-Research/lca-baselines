  ```
import os
import json
from pyaedt import Hfss

# Set non-graphical mode
non_graphical = False

# Download 3D component
component_path = "path/to/3D/component.aedt"

# Launch HFSS and save project with unique name
hfss = Hfss(non_graphical=non_graphical)
hfss.save_project(generate_unique_name("MyProject"))

# Read array definition from JSON file
with open("array_definition.json") as f:
    array_definition = json.load(f)

# Load 3D component into dictionary from specified path
component = hfss.download_icepak_3d_component(component_path)

# Set up simulation and analyze
hfss.modeler.create_array(array_definition, component)
hfss.analyze_nominal()

# Get far field data after simulation completes
far_field_data = hfss.get_far_field_data()

# Generate contour plot
hfss.post.create_contour_plot(far_field_data)

# Generate 2D cutout plots
hfss.post.create_cutout_plot(far_field_data)

# Generate 3D polar plots in Matplotlib
hfss.post.create_polar_plot(far_field_data)

# Generate 3D plots in PyVista
hfss.post.create_3d_plot(far_field_data)

# Release AEDT
hfss.release_desktop()
```