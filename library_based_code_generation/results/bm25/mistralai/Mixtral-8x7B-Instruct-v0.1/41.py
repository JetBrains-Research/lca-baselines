 ```python
import metpy.calc as mpcalc
import metpy.plots as mpplots
import metpy.units as mpunits
import xarray as xr

# Load sample dataset
ds = xr.tutorial.load_dataset('sounding')

# Drop rows with NaN values in specific columns
ds = ds.dropna(dim='station', how='any')

# Assign units
ds['pressure'] = ds['pressure'] * mpunits.hPa
ds['temperature'] = ds['temperature'] * mpunits.degC
ds['dewpoint'] = ds['dewpoint'] * mpunits.degC

# Create new figure with specific aspect ratio
fig = mpplots.Fig(figsize=(10, 10))

# Plot data using normal plotting functions
ax = fig.add_subplot(111)
mpplots.sounding(ax, ds, 'pressure', ['temperature', 'dewpoint'])

# Set custom labels for the x and y axes
ax.set_xlabel('Pressure (hPa)')
ax.set_ylabel('Temperature (degC) / Dewpoint (degC)')

# Calculate LCL and plot it as a black dot
lcl = mpcalc.lcl(ds['temperature'], ds['dewpoint'], ds['pressure'])
ax.plot(lcl.pressure, lcl.temperature, 'bo')

# Calculate full parcel profile and add it to the plot as a black line
parcel = mpcalc.parcel_profile(ds['pressure'], ds['temperature'].values[0], ds['dewpoint'].values[0])
ax.plot(parcel.pressure, parcel.temperature, 'k-')

# Shade areas of CAPE and CIN
cape, cin, _ = mpcalc.cape_cin(parcel.pressure, parcel.temperature, parcel.dewpoint, ds['pressure'], ds['temperature'], ds['dewpoint'])
ax.fill_betweenx(parcel.pressure, parcel.temperature, parcel.temperature + cin, alpha=0.3, color='red')
ax.fill_betweenx(parcel.pressure, parcel.temperature + cape, parcel.temperature + 10, alpha=0.3, color='green')

# Add special lines to the plot
mpplots.held_plot(ax, ds.pressure, ds.temperature, ds.dewpoint)

# Display the plot
fig.show()
```