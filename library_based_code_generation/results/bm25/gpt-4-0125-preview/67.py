```python
from pyaedt import Hfss, Desktop
import os

# Set non-graphical mode
non_graphical = os.getenv("PYAEDT_NON_GRAPHICAL")

# Launch AEDT in non-graphical mode
desktop = Desktop(specified_version="2023.1", non_graphical=non_graphical)

# Create or open a project in 2D Extractor
project_name = "CPWG_Design"
project_path = os.path.join(desktop.projects_path, project_name + ".aedt")
hfss = Hfss(projectname=project_path, designname="CPWG_2D_Extractor", solution_type="Terminal")

# Define variables
hfss["substrate_height"] = "1.6mm"
hfss["signal_width"] = "0.5mm"
hfss["gap_width"] = "0.2mm"
hfss["ground_width"] = "5mm"
hfss["dielectric_constant"] = "4.4"
hfss["conformal_coating_thickness"] = "0.1mm"

# Create primitives
substrate = hfss.modeler.create_box([0, 0, 0], [hfss["ground_width"], hfss["ground_width"], -hfss["substrate_height"]], name="Substrate", matname="FR4")
signal = hfss.modeler.create_rectangle(hfss.PLANE.XY, [hfss["gap_width"], 0, 0], [hfss["signal_width"], hfss["ground_width"]], name="Signal", matname="copper")
ground_left = hfss.modeler.create_rectangle(hfss.PLANE.XY, [0, 0, 0], [hfss["gap_width"], hfss["ground_width"]], name="GroundLeft", matname="copper")
ground_right = hfss.modeler.create_rectangle(hfss.PLANE.XY, [2*hfss["gap_width"] + hfss["signal_width"], 0, 0], [hfss["gap_width"], hfss["ground_width"]], name="GroundRight", matname="copper")

# Create a dielectric
dielectric = hfss.modeler.create_box([0, 0, -hfss["substrate_height"]], [hfss["ground_width"], hfss["ground_width"], -hfss["substrate_height"]], name="Dielectric", matname="FR4")

# Create a conformal coating
conformal_coating = hfss.modeler.create_box([0, 0, 0], [hfss["ground_width"], hfss["ground_width"], hfss["conformal_coating_thickness"]], name="ConformalCoating", matname="Silicon")

# Assign a conductor to the signal
hfss.assign_perfect_e(signal.name)

# Create a reference ground
hfss.assign_perfect_e([ground_left.name, ground_right.name])

# Assign the Huray model on the signal
hfss.assign_material(signal.name, "copper", "Huray")

# Create the setup
setup = hfss.create_setup(name="Setup1")
setup.props["Frequency"] = "5GHz"
setup.props["MaximumPasses"] = 15
setup.props["MinimumPasses"] = 2
hfss.create_linear_count_sweep(setupname="Setup1", unit="GHz", freqstart=1, freqstop=10, num_of_freq_points=101, sweepname="Sweep1", sweep_type="Interpolating")

# Analyze
hfss.analyze_all()

# Plot solution data
hfss.post.create_report("S Parameters", "Setup1 : Sweep1", ["Freq"], ["dB(S(1,1))", "dB(S(2,1))"])

# Save the project
hfss.save_project()

# Close AEDT
desktop.release_desktop()
```