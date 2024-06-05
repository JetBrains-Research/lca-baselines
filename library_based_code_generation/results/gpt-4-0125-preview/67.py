```python
from pyaedt import Hfss, Desktop
import os

# Set non-graphical mode and launch AEDT with 2D Extractor
desktop_version = "2023.1"
non_graphical = True
desktop = Desktop(desktop_version, non_graphical, False)
aedt_app = Hfss(specified_version=desktop_version, design_type="2D Extractor")

# Define variables
width = "0.2mm"
gap = "0.1mm"
length = "10mm"
thickness = "0.035mm"
dielectric_thickness = "0.5mm"
conformal_coating_thickness = "0.1mm"
material = "copper"
dielectric_material = "FR4"
conformal_coating_material = "solder_mask"

# Create primitives
oEditor = aedt_app.modeler.oeditor
oEditor.CreateRectangle(
    [
        "NAME:RectangleParameters",
        "IsCovered:=", True,
        "XStart:=", "-width/2-gap",
        "YStart:=", "0mm",
        "ZStart:=", "0mm",
        "Width:=", width,
        "Height:=", length,
        "WhichAxis:=", "Z"
    ],
    [
        "NAME:Attributes",
        "Name:=", "Signal",
        "Flags:=", "",
        "Color:=", "(132 132 193)",
        "Transparency:=", 0,
        "PartCoordinateSystem:=", "Global",
        "UDMId:=", "",
        "MaterialValue:=", "\"{}\"".format(material),
        "SolveInside:=", True
    ]
)

# Create coplanar ground
oEditor.DuplicateAroundAxis(
    [
        "NAME:Selections",
        "Selections:=", "Signal",
        "NewPartsModelFlag:=", "Model"
    ], 
    [
        "NAME:DuplicateAroundAxisParameters",
        "CreateNewObjects:=", True,
        "WhichAxis:=", "Y",
        "AngleStr:=", "180deg",
        "NumClones:=", "1"
    ]
)

# Create reference ground plane
oEditor.CreateRectangle(
    [
        "NAME:RectangleParameters",
        "IsCovered:=", True,
        "XStart:=", "-3*width",
        "YStart:=", "0mm",
        "ZStart:=", "-dielectric_thickness-thickness",
        "Width:=", "6*width",
        "Height:=", length,
        "WhichAxis:=", "Z"
    ],
    [
        "NAME:Attributes",
        "Name:=", "GroundPlane",
        "Flags:=", "",
        "Color:=", "(132 132 193)",
        "Transparency:=", 0,
        "PartCoordinateSystem:=", "Global",
        "UDMId:=", "",
        "MaterialValue:=", "\"{}\"".format(material),
        "SolveInside:=", True
    ]
)

# Create dielectric
oEditor.CreateBox(
    [
        "NAME:BoxParameters",
        "XPosition:=", "-3*width",
        "YPosition:=", "0mm",
        "ZPosition:=", "-dielectric_thickness",
        "XSize:=", "6*width",
        "YSize:=", length,
        "ZSize:=", dielectric_thickness
    ],
    [
        "NAME:Attributes",
        "Name:=", "Dielectric",
        "Flags:=", "",
        "Color:=", "(143 175 143)",
        "Transparency:=", 0,
        "PartCoordinateSystem:=", "Global",
        "UDMId:=", "",
        "MaterialValue:=", "\"{}\"".format(dielectric_material),
        "SolveInside:=", True
    ]
)

# Create conformal coating
oEditor.CreateBox(
    [
        "NAME:BoxParameters",
        "XPosition:=", "-width/2-gap",
        "YPosition:=", "0mm",
        "ZPosition:=", "0mm",
        "XSize:=", "2*(width+gap)",
        "YSize:=", length,
        "ZSize:=", conformal_coating_thickness
    ],
    [
        "NAME:Attributes",
        "Name:=", "ConformalCoating",
        "Flags:=", "",
        "Color:=", "(255 255 0)",
        "Transparency:=", 0,
        "PartCoordinateSystem:=", "Global",
        "UDMId:=", "",
        "MaterialValue:=", "\"{}\"".format(conformal_coating_material),
        "SolveInside:=", True
    ]
)

# Assign conductor to the signal
aedt_app.assign_material("Signal", material)

# Create reference ground
aedt_app.assign_perfect_e(["GroundPlane"])

# Assign the Huray model on the signal
aedt_app.assign_surface_roughness("Signal", model_type="Huray", roughness="1.4um")

# Create the setup
setup = aedt_app.create_setup(setupname="MySetup")
setup.props["Frequency"] = "5GHz"
setup.props["MaximumPasses"] = 10
setup.props["MinimumPasses"] = 1
setup.update()

# Analyze
aedt_app.analyze_setup("MySetup")

# Plot solution data
aedt_app.post.create_report(expression="S(1,1)", domain="Sweep")

# Save the project and close AEDT
project_path = os.path.join(aedt_app.working_directory, "CPWG_Design.aedt")
aedt_app.save_project(project_path)
aedt_app.close_desktop()
```