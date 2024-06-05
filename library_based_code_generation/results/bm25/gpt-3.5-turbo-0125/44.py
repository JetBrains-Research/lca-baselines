import os
from pyaedt import Hfss3dLayout, Desktop
from pyaedt import Q3d

project_path = os.path.join(os.getcwd(), "Project")
project_name = "PCB_Analysis"
hfss_layout = Hfss3dLayout(projectname=project_name, projectpath=project_path)
hfss_layout.import_edb("path_to_edb_file")
hfss_layout.create_cutout_on_selected_nets(["net1", "net2"])
hfss_layout.export_to_q3d("path_to_q3d_file")
pin_locations = {"component1": (x1, y1, z1), "component2": (x2, y2, z2)}
hfss_layout.assign_sources_and_sinks(pin_locations)
hfss_layout.save_project()
hfss_layout.close_project()

hfss_layout.open_in_hfss_3dlayout()
hfss_layout.export_to_q3d("path_to_q3d_file")
q3d = Q3d()
q3d.open_project("path_to_q3d_file")
q3d.plot()
q3d.assign_sources_and_sinks(pin_locations)
q3d.create_setup_and_frequency_sweep(start_freq="DC", stop_freq="2GHz")
q3d.analyze()
q3d.plot_acl()
q3d.plot_acr()
q3d.release_desktop()