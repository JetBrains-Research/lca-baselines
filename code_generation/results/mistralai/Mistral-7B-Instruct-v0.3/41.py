 ```python
import metpy.calc as mpcalc
import metpy.unitful as u
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Load sample dataset (assuming it's a Dataset object from xarray)
data = xr.tutorial.load_example_data('air_profile')

# Clean the data by dropping rows with NaN values in specific columns
data = data.dropna(subset=['t', 'p'])

# Assign units to the data
data['t'].units = u.degC
data['p'].units = u.Pa
data['z'].units = u.m

# Create a new figure with a specific aspect ratio
fig, ax = plt.subplots(figsize=(10, 6), aspect='auto')

# Plot the data using normal plotting functions
ax.plot(data['z'], data['t'], label='Temperature')
ax.plot(data['z'], data['p'], label='Pressure')

# Calculate LCL and plot it as a black dot
lcl = mpcalc.lcl(data['t'].values, data['p'].values, data['q'].values)
ax.plot(lcl['z'].m, lcl['t'].C, 'ko', markersize=10, label='LCL')

# Calculate full parcel profile and add it to the plot as a black line
parcel = mpcalc.parcel_profile(data['t'].values, data['p'].values, data['q'].values, data['z'].values)
ax.plot(parcel['z'].m, parcel['t'].C, 'k--', label='Full Parcel Profile')

# Calculate CAPE and CIN and shade the areas
cape = mpcalc.cape(data['t'].values, data['p'].values, data['q'].values, data['z'].values)
cin = mpcalc.cin(data['t'].values, data['p'].values, data['q'].values, data['z'].values)
ax.fill_between(data['z'].values, cape.m, 0, color='green', alpha=0.3, label='CAPE')
ax.fill_between(data['z'].values, 0, cin.m, color='red', alpha=0.3, label='CIN')

# Add special lines to the plot
ax.axhline(0, color='black', linestyle='--', label='Surface')
ax.axhline(data['z'].values[-1], color='black', linestyle='--', label='Top of the atmosphere')

# Set custom labels for the x and y axes
ax.set_xlabel('Altitude (m)')
ax.set_ylabel('Temperature (°C)')

# Display the plot
plt.legend()
plt.show()
```