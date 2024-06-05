import pyaedt

# Import necessary libraries
from pyaedt import CreateAedtObj, CreateAedtApplication

# Set non-graphical mode
pyaedt.init("", False)

# Launch AEDT and 2D Extractor
aedtapp = CreateAedtApplication()
aedtapp.create_2dextractor()

# Define variables
width = 1
height = 0.5
thickness = 0.02
dielectric_thickness = 0.01

# Create primitives
signal = aedtapp.modeler.primitives.create_rectangle([0, 0, 0], [width, height], name="Signal")
ground = aedtapp.modeler.primitives.create_rectangle([0, -height, 0], [width, 0], name="Ground")
dielectric = aedtapp.modeler.primitives.create_rectangle([0, -height, 0], [width, -height+dielectric_thickness], name="Dielectric")

# Create signal
signal = aedtapp.modeler.assign_conductor(signal, "Signal")

# Create reference ground plane
reference_ground = aedtapp.modeler.primitives.create_rectangle([0, -height-thickness, 0], [width, -height], name="ReferenceGround")

# Create conformal coating
conformal_coating = aedtapp.modeler.primitives.create_rectangle([0, -height-dielectric_thickness, 0], [width, -height-thickness], name="ConformalCoating")

# Assign Huray model on the signal
aedtapp.assign_huray_model(signal)

# Create setup
setup = aedtapp.create_setup("MySetup")
setup.props["Freq"] = "5GHz"
setup.props["MaxPass"] = 10

# Analyze
aedtapp.analyze_nominal()

# Plot solution data
aedtapp.plot_fields("MySetup", "db(S(1,1))")

# Save project
aedtapp.save_project("MyCPWGDesign.aedt")

# Close AEDT
aedtapp.close_project()