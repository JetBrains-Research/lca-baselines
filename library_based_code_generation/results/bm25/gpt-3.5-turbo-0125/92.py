import os
from pyaedt import Hfss
from pyaedt import Desktop
from pyaedt import generate_unique_name, generate_unique_project_name, generate_unique_folder_name

project_name = generate_unique_project_name()
project_dir = generate_unique_folder_name()

hfss = Hfss()
hfss.save_project(os.path.join(project_dir, project_name))

hfss.download_icepak_3d_component("component_name")

hfss.add_3d_component_array_from_json("array_definition.json", "component_path")

hfss.insert_3d_component("component_name")

hfss.analyze_nominal()

far_field_data = hfss.get_far_field_data()

hfss.plot_contour()

hfss.plot_cutout_2d()

hfss.plot_3d_polar_matplotlib()

hfss.plot_3d_pyvista()

Desktop.ReleaseAEDT()