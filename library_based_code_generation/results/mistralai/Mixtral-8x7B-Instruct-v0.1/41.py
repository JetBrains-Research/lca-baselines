 ```python
import metpy.calc as mpcalc
import metpy.plots as mpplots
import metpy.units as mpunits
import xarray as xr

# Load sample dataset
ds = xr.tutorial.load_dataset('era5_example')

# Clean data by dropping rows with NaN values in specific columns
clean_data = ds.dropna(dim='step', how='any')

# Assign units to the data
latitude = clean_data['latitude'] * mpunits.degrees
longitude = clean_data['longitude'] * mpunits.degrees
time = clean_data['time'] * mpunits.days
height = clean_data['height'] * mpunits.meters
temperature = clean_data['temperature'] * mpunits.kelvin
dewpoint = clean_data['dewpoint_temperature'] * mpunits.kelvin
u_wind = clean_data['u_component_of_wind'] * mpunits.meters_per_second
v_wind = clean_data['v_component_of_wind'] * mpunits.meters_per_second

# Create new figure with specific aspect ratio
fig = mpplots.Fig(figsize=(10, 8), dpi=100)

# Plot data using normal plotting functions
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
mpplots.contourf(time, height, temperature, ax=ax, cmap='coolwarm', units='K')
mpplots.contour(time, height, dewpoint, ax=ax, colors='black', units='K')
mpplots.plot(time, height, '--', color='black', ax=ax)
mpplots.plot(time, height, ':', color='black', ax=ax)

# Set custom labels for the x and y axes
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Height (m)')

# Calculate LCL and plot it as a black dot
lcl = mpcalc.lcl(temperature, dewpoint)
ax.plot(time, lcl, 'ko', ms=2)

# Calculate full parcel profile and add it to the plot as a black line
parcel_profile = mpcalc.parcel_profile(temperature, dewpoint, height)
ax.plot(time, parcel_profile, 'k-', lw=1.5)

# Shade areas of CAPE and CIN
shaded_area = mpplots.cape_cin_shading(time, height, temperature, dewpoint, u_wind, v_wind)
ax.add_geometries(shaded_area, 'polygon', facecolor='lightgrey', edgecolor='none')

# Add special lines to the plot
mpplots.sounding_winds(time, height, u_wind, v_wind, ax=ax)
mpplots.pressure_coordinates(height, ax=ax, units=mpunits.hectopascals)

# Display the plot
mpplots.show(fig)
```