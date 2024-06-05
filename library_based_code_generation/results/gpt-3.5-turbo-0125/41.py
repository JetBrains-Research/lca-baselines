import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metpy.calc import lcl, parcel_profile, cape_cin
from metpy.plots import SkewT
from metpy.units import units

# Load sample dataset
data = pd.read_csv('sample_data.csv')

# Clean data
data_cleaned = data.dropna(subset=['temperature', 'dewpoint'])

# Assign units
data_cleaned['pressure'] = data_cleaned['pressure'].values * units.hPa
data_cleaned['temperature'] = data_cleaned['temperature'].values * units.degC
data_cleaned['dewpoint'] = data_cleaned['dewpoint'].values * units.degC

# Create figure
fig = plt.figure(figsize=(10, 10))
skew = SkewT(fig)

# Plot data
skew.plot(data_cleaned['pressure'], data_cleaned['temperature'], 'r')
skew.plot(data_cleaned['pressure'], data_cleaned['dewpoint'], 'g')

# Custom labels
plt.xlabel('Temperature (C)')
plt.ylabel('Pressure (hPa)')

# Calculate LCL
lcl_pressure, lcl_temperature = lcl(data_cleaned['pressure'][0], data_cleaned['temperature'][0], data_cleaned['dewpoint'][0])
skew.plot(lcl_pressure, lcl_temperature, 'ko', markersize=10)

# Calculate parcel profile
prof = parcel_profile(data_cleaned['pressure'], data_cleaned['temperature'][0], data_cleaned['dewpoint'][0])
skew.plot(data_cleaned['pressure'], prof, 'k')

# Calculate CAPE and CIN
cape, cin = cape_cin(data_cleaned['pressure'], data_cleaned['temperature'], data_cleaned['dewpoint'])
skew.shade_cin(data_cleaned['pressure'], data_cleaned['temperature'], prof)
skew.shade_cape(data_cleaned['pressure'], data_cleaned['temperature'], prof)

# Add special lines
skew.ax.axvline(0, color='b', linestyle='--', linewidth=2)
skew.ax.axvline(-20, color='b', linestyle='--', linewidth=2)

# Display plot
plt.show()