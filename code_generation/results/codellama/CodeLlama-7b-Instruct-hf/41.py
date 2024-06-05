  import numpy as np
import matplotlib.pyplot as plt
from metpy.units import units
from metpy.plots import SkewT
from metpy.calc import lcl, full_parcel

# Load sample dataset
data = np.loadtxt('sounding_data.txt')

# Drop rows with NaN values in specific columns
data = data[~np.isnan(data[:, 1:]).any(axis=1)]

# Assign units to the data
data[:, 1:] = data[:, 1:].astype(units.Quantity)

# Create new figure with specific aspect ratio
fig, ax = plt.subplots(figsize=(10, 8), aspect=1.5)

# Plot data using normal plotting functions
skew = SkewT(ax, data[:, 0], data[:, 1], data[:, 2])
skew.plot(data[:, 3], data[:, 4], 'bo')

# Set custom labels for x and y axes
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Dew Point (K)')

# Calculate LCL and plot it as a black dot
lcl_level = lcl(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
skew.plot(lcl_level, data[:, 2], 'ko')

# Calculate full parcel profile and add it to the plot as a black line
full_parcel_profile = full_parcel(data[:, 1], data[:, 2], data[:, 3], data[:, 4])
skew.plot(full_parcel_profile, data[:, 2], 'k-')

# Shade areas of CAPE and CIN
skew.shade_cape()
skew.shade_cin()

# Add special lines to the plot
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Display the plot
plt.show()