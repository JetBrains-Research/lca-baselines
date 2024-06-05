  import pyaedt

# Set non-graphical mode
non_graphical = True

# Launch AEDT in specified version and solution type
aedtapp = pyaedt.Hfss(specified_version, "HFSS")

# Set some properties
aedtapp.modeler.model_units = "mm"
aedtapp.modeler.auto_generate_causal_materials = True
aedtapp.modeler.auto_generate_causal_dimensions = True
aedtapp.modeler.auto_generate_causal_objects = True
aedtapp.modeler.auto_generate_causal_ports = True

# Define variables for flex cable CPWG
total_length = 100e-3
theta = 0
radius = 50e-3
width = 10e-3
height = 10e-3
spacing = 10e-3
ground_width = 10e-3
ground_thickness = 10e-3

# Define function to create bending based on curvature radius and extension
def create_bending(radius, extension):
    return aedtapp.modeler.create_circle(radius, [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1])

# Draw signal line and ground line
signal_line = aedtapp.modeler.create_line([0, 0, 0], [total_length, 0, 0], width, "Copper")
ground_line = aedtapp.modeler.create_line([0, 0, 0], [total_length, 0, 0], ground_width, "Ground")

# Draw dielectric to create dielectric cable
dielectric_cable = aedtapp.modeler.create_box([0, 0, 0], [total_length, height, width], "Air")

# Create bottom metals
bottom_metals = aedtapp.modeler.create_box([0, 0, 0], [total_length, height, width], "Copper")

# Create port interfaces (PEC enclosures) and Perfect E boundary condition
aedtapp.modeler.create_pec_enclosure(signal_line, "PEC_1")
aedtapp.modeler.create_pec_enclosure(ground_line, "PEC_2")
aedtapp.modeler.create_perfect_e_boundary(signal_line, "PEC_1")
aedtapp.modeler.create_perfect_e_boundary(ground_line, "PEC_2")

# Create ports and setup and sweep with specified properties
aedtapp.modeler.create_port(signal_line, "Port_1")
aedtapp.modeler.create_port(ground_line, "Port_2")
aedtapp.set_active_editor("Layout")
aedtapp.create_setup(solution_type="EigenMode", setup_name="Setup1")
aedtapp.create_sweep_linear(setup_name="Setup1", field="Ex", freq_start=1e9, freq_stop=100e9, freq_step=1e9)

# Plot model and release AEDT
aedtapp.plot(show=False, export_path="C:\\temp\\")
aedtapp.release_desktop()