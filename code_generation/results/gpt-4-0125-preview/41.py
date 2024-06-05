```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metpy.calc import dewpoint_from_relative_humidity, lcl, parcel_profile, cape_cin
from metpy.plots import SkewT
from metpy.units import units

# Load a sample dataset
df = pd.read_csv('sample_sounding_data.csv')

# Clean the dataset by dropping rows with NaN values in specific columns
df = df.dropna(subset=['pressure', 'temperature', 'dewpoint', 'relative_humidity'])

# Assign units to the data
pressure = df['pressure'].values * units.hPa
temperature = df['temperature'].values * units.degC
dewpoint = dewpoint_from_relative_humidity(temperature, df['relative_humidity'].values / 100)
relative_humidity = df['relative_humidity'].values * units.percent

# Create a new figure with a specific aspect ratio
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

# Plot the data
skew.plot(pressure, temperature, 'r', label='Temperature')
skew.plot(pressure, dewpoint, 'g', label='Dew Point')

# Set custom labels for the x and y axes
skew.ax.set_xlabel('Temperature (°C)')
skew.ax.set_ylabel('Pressure (hPa)')

# Calculate the LCL and plot it as a black dot
lcl_pressure, lcl_temperature = lcl(pressure[0], temperature[0], dewpoint[0])
skew.plot(lcl_pressure, lcl_temperature, 'ko', label='LCL')

# Calculate the full parcel profile and add it to the plot as a black line
parcel_prof = parcel_profile(pressure, temperature[0], dewpoint[0]).to('degC')
skew.plot(pressure, parcel_prof, 'k', label='Parcel Profile')

# Shade areas of CAPE and CIN
skew.shade_cape(pressure, temperature, parcel_prof)
skew.shade_cin(pressure, temperature, parcel_prof, dewpoint)

# Add special lines to the plot (dry adiabats, moist adiabats, and mixing ratio lines)
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Add a legend
plt.legend()

# Display the plot
plt.show()
```