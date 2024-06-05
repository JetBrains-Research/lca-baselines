  import MetPy
import numpy as np
import matplotlib.pyplot as plt

# Load sample dataset
ds = MetPy.sample_data.load_sounding('ERAI')

# Clean the data by dropping rows with NaN values in specific columns
ds = ds.dropna(subset=['pressure', 'temperature', 'dewpoint'])

# Assign units to the data
ds['pressure'] = ds['pressure'].to_units('Pa')
ds['temperature'] = ds['temperature'].to_units('degC')
ds['dewpoint'] = ds['dewpoint'].to_units('degC')

# Create a new figure with a specific aspect ratio
fig, ax = plt.subplots(figsize=(10, 6), aspect=1.5)

# Plot the data using normal plotting functions
ax.plot(ds['pressure'], ds['temperature'], label='Temperature')
ax.plot(ds['pressure'], ds['dewpoint'], label='Dewpoint')

# Calculate the lifted condensation level (LCL) and plot it as a black dot
lcl = MetPy.calc.lcl(ds['pressure'], ds['temperature'], ds['dewpoint'])
ax.plot(lcl, ds['temperature'][lcl], 'ko')

# Calculate the full parcel profile and add it to the plot as a black line
parcel = MetPy.calc.parcel_profile(ds['pressure'], ds['temperature'], ds['dewpoint'])
ax.plot(parcel['pressure'], parcel['temperature'], 'k-')

# Shade areas of Convective Available Potential Energy (CAPE) and Convective Inhibition (CIN)
cape = MetPy.calc.cape(ds['pressure'], ds['temperature'], ds['dewpoint'])
cin = MetPy.calc.cin(ds['pressure'], ds['temperature'], ds['dewpoint'])
ax.fill_between(ds['pressure'], ds['temperature'], cape, color='#ff7f0e', label='CAPE')
ax.fill_between(ds['pressure'], ds['temperature'], cin, color='#2ca02c', label='CIN')

# Add special lines to the plot and display it
ax.axhline(0, color='k', linestyle='--')
ax.axvline(lcl, color='k', linestyle='--')
ax.legend()
plt.show()