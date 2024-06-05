  ```
import pyaedt

# Set non-graphical mode
non_graphical = False

# Launch AEDT in specified version and solution type
aedt_app = pyaedt.Hfss(specified_version, "HFSS")

# Set properties
aedt_app.modeler.model_units = "mm"
aedt_app.modeler.initial_mesh_size = "0.01mm"
aedt_app.modeler.auto_causal_materials = True

# Define variables for flex cable CPWG
total_length = 100
theta = 0
radius = 50
width = 10
height = 10
spacing = 10
ground_width = 10
ground_thickness = 10

# Define function to create bending
def create_bending(curvature_radius, extension):
    return aedt_app.modeler.create_bend(curvature_radius, extension)

# Draw signal line and ground line
signal_line = aedt_app.modeler.create_polyline(
    [0, 0], [total_length, 0], width, height, spacing
)
ground_line = aedt_app.modeler.create_polyline(
    [0, 0], [total_length, 0], ground_width, ground_thickness, spacing
)

# Draw dielectric cable
dielectric_cable = aedt_app.modeler.create_polyline(
    [0, 0], [total_length, 0], width, height, spacing
)

# Create bottom metals
bottom_metals = aedt_app.modeler.create_polyline(
    [0, 0], [total_length, 0], width, height, spacing
)

# Create port interfaces (PEC enclosures)
aedt_app.modeler.create_port_interface(
    "PEC", [0, 0], [total_length, 0], width, height, spacing
)

# Create Perfect E boundary condition
aedt_app.modeler.create_perfect_e_boundary(
    [0, 0], [total_length, 0], width, height, spacing
)

# Create ports and setup and sweep
aedt_app.modeler.create_port(
    [0, 0], [total_length, 0], width, height, spacing
)
aedt_app.modeler.create_setup(
    "Setup1", "FrequencySweep", "1GHz", "10GHz", 101
)
aedt_app.modeler.create_sweep(
    "Sweep1", "Freq", "1GHz", "10GHz", 101
)

# Plot model and release AEDT
aedt_app.plot(show=False)
aedt_app.release_desktop()
```