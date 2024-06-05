 Here is the Python code that follows your instructions using the MetPy library:

```python
import metpy.calc as mpcalc
import metpy.io as io
import metpy.units as u
import matplotlib.pyplot as plt
import xarray as xr

# Load a sample dataset (e.g., SKEW-T diagram)
data = io.read_file('sounding.txt')

# Clean the data by dropping rows with NaN values in specific columns
data = data.dropna(subset=['temperature', 'pressure', 'relative_humidity'])

# Assign units to the data
data['temperature'].units = u.kelvin
data['pressure'].units = u.Pa
data['relative_humidity'].units = u.dimensionless_unscaled

# Create a new figure with a specific aspect ratio
fig, ax = plt.subplots(figsize=(10, 6), aspect='equal')

# Plot the data using normal plotting functions
data.plot(x='pressure', y='temperature', kind='line', ax=ax)

# Set custom labels for the x and y axes
ax.set_xlabel('Pressure (Pa)')
ax.set_ylabel('Temperature (K)')

# Calculate the LCL and plot it as a black dot
lcl = mpcalc.lifted_condensation_level(data['temperature'], data['relative_humidity'], data['pressure'])
ax.plot(lcl.pressure, lcl.temperature, 'ko')

# Calculate the full parcel profile and add it to the plot as a black line
parcel = mpcalc.parcel_profile(data['temperature'], data['relative_humidity'], data['pressure'])
ax.plot(parcel.pressure, parcel.temperature, 'k--')

# Calculate and shade areas of Convective Available Potential Energy (CAPE) and Convective Inhibition (CIN)
cape = mpcalc.convective_available_potential_energy(data['temperature'], data['dewpoint'], data['pressure'])
cin = mpcalc.convective_inhibition(data['temperature'], data['dewpoint'], data['pressure'])
ax.fill_between(data['pressure'], cape, 0, color='green', alpha=0.5)
ax.fill_between(data['pressure'], cin, 0, color='red', alpha=0.5)

# Add special lines to the plot
ax.axhline(data['temperature'].values[0], color='black', linestyle='--')  # Dry adiabat
ax.axhline(data['dewpoint'].values[0], color='black', linestyle='--')  # Wet adiabat
ax.axhline(data['temperature'].values[-1], color='black', linestyle='--')  # Environment temperature
ax.axhline(data['dewpoint'].values[-1], color='black', linestyle='--')  # Environment dewpoint

# Display the plot
plt.show()
```

This code assumes that the sounding.txt file contains temperature, pressure, and relative humidity data in the format expected by the MetPy `io.read_file()` function. You may need to adjust the code to fit your specific dataset.