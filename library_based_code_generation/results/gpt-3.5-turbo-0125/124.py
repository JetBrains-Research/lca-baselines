import os
from pyaedt import generate_unique_name

temp_folder = os.path.join(os.getcwd(), generate_unique_name("temp"))
print("Temporary folder path:", temp_folder)

# Download example file into the temporary folder

# Set non-graphical mode and launch AEDT in graphical mode using SI units

# Initialize AEDT and launch HFSS 3D Layout

# Remove AEDT file if it already exists and save project in temporary folder

# Print boundaries from setups object

# Hide all nets and make only two specified nets visible

# Plot the two specified nets

# Make all layers visible

# Change color of specified layer

# Disable visibility of components for top and bottom layers

# Fit all to visualize all components

# Close project and release desktop