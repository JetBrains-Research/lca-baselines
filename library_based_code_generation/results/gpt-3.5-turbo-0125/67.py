import os
from pyaedt import Hfss
from pyaedt import Desktop
from pyaedt import Cpws
from pyaedt import Constants

project_name = "CPWG_Design"
design_name = "CPWG_Design"

hfss = Hfss()
hfss.save_project(os.path.join(hfss.working_directory, project_name + ".aedt"))

hfss.create_variable("Width", "1mm")
hfss.create_variable("Height", "0.5mm")
hfss.create_variable("Length", "10mm")
hfss.create_variable("Gap", "0.2mm")
hfss.create_variable("Substrate_Height", "1mm")
hfss.create_variable("Conductor_Thickness", "0.035mm")
hfss.create_variable("Dielectric_Constant", "4.2")
hfss.create_variable("Conformal_Coating_Thickness", "0.01mm")

hfss.create_cpwg("CPWG", "Width", "Height", "Length", "Gap", "Substrate_Height", "Conductor_Thickness", "Dielectric_Constant", "Conformal_Coating_Thickness")

hfss.create_signal("Signal", "CPWG")
hfss.create_coplanar_ground("Ground", "CPWG")
hfss.create_reference_ground_plane("Reference_Ground", "CPWG")
hfss.create_dielectric("Dielectric", "CPWG")
hfss.create_conformal_coating("Conformal_Coating", "CPWG")
hfss.assign_conductor("Signal", "CPWG")
hfss.create_reference_ground("Reference_Ground", "CPWG")
hfss.assign_huray_model("Signal", "CPWG")

hfss.create_setup("Setup", 1e9)
hfss.analyze_setup("Setup")
hfss.plot_fields("Setup")
hfss.save_project()
hfss.close_project()