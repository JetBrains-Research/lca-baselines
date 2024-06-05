```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metpy.calc import dewpoint_from_relative_humidity, lcl, parcel_profile, cape_cin
from metpy.plots import SkewT
from metpy.units import units
from metpy.io import get_upper_air_data
from datetime import datetime

# Load a sample dataset
date = datetime(2020, 5, 3, 0)  # Example date
station = 'OUN'  # Example station
df = get_upper_air_data(date, station)

# Clean the dataset by dropping rows with NaN values in 'pressure', 'temperature', 'dewpoint'
df = df.dropna(subset=['pressure', 'temperature', 'dewpoint'])

# Assign units to the data
pressure = df['pressure'].values * units.hPa
temperature = df['temperature'].values * units.degC
dewpoint = df['dewpoint'].values * units.degC

# Create a new figure with a specific aspect ratio
fig = plt.figure(figsize=(10, 10))
skew = SkewT(fig, rotation=45)

# Plot the data
skew.plot(pressure, temperature, 'r', label='Temperature')
skew.plot(pressure, dewpoint, 'g', label='Dewpoint')

# Set custom labels for the x and y axes
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (hPa)')

# Calculate the LCL and plot it as a black dot
lcl_pressure, lcl_temperature = lcl(pressure[0], temperature[0], dewpoint[0])
skew.plot(lcl_pressure, lcl_temperature, 'ko', label='LCL')

# Calculate the full parcel profile and add it to the plot as a black line
prof = parcel_profile(pressure, temperature[0], dewpoint[0]).to('degC')
skew.plot(pressure, prof, 'k', linewidth=2, label='Parcel Profile')

# Shade areas of CAPE and CIN
skew.shade_cape(pressure, temperature, prof)
skew.shade_cin(pressure, temperature, prof, dewpoint)

# Add special lines to the plot (dry adiabats, moist adiabats, and mixing ratio lines)
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Display the plot
plt.legend(loc='best')
plt.show()
```