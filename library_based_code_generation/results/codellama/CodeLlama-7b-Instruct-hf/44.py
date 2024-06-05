  import os
import PyAEDT

# Set up project files and paths
project_file = "project.aedt"
project_path = "C:\\path\\to\\project"
temp_project_path = "C:\\path\\to\\temp\\project"

# Download project file
os.system(f"curl -o {project_file} https://www.example.com/project.aedt")

# Set up temporary project directory
os.system(f"mkdir {temp_project_path}")

# Open EDB project
edb = PyAEDT.Edb(project_file, project_path, temp_project_path)

# Create cutout on selected nets
edb.create_cutout(["Net1", "Net2"])

# Export to Q3D
edb.export_to_q3d()

# Identify pin locations on components to define sources and sinks
pin_locations = edb.get_pin_locations()

# Append Z elevation to pin locations
pin_locations = [pin_location + (0, 0, 10) for pin_location in pin_locations]

# Save and close EDB
edb.save_project()
edb.close_project()

# Open EDB in Hfss 3D Layout
hfss = PyAEDT.Hfss(project_file, project_path, temp_project_path)

# Generate 3D model
hfss.generate_3d_model()

# Export layout to Q3D
hfss.export_to_q3d()

# Launch Q3D project
q3d = PyAEDT.Q3d(project_file, project_path, temp_project_path)

# Plot project
q3d.plot()

# Assign sources and sinks on nets using previously calculated positions
q3d.assign_sources_and_sinks(pin_locations)

# Create setup and frequency sweep
q3d.create_setup("Setup1")
q3d.create_frequency_sweep(1e9, 2e9, 100)

# Analyze project
q3d.analyze_project()

# Compute ACL and ACR solutions
q3d.compute_acl_and_acr()

# Plot ACL and ACR solutions
q3d.plot_acl_and_acr()

# Release desktop
q3d.release_desktop()