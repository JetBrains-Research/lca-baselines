```python
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.io import get_upper_air_data
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir

def effective_inflow_layer(sounding, cape_threshold=100 * units('J/kg'), cin_threshold=-250 * units('J/kg')):
    """
    Determine the effective inflow layer for a convective sounding.
    Default values based on Thompson et al. (2004) for CAPE and CIN.
    """
    # Extract pressure and temperature
    pressure = sounding['pressure'].values * units(sounding['pressure'].units)
    temperature = sounding['temperature'].values * units(sounding['temperature'].units)
    dewpoint = sounding['dewpoint'].values * units(sounding['dewpoint'].units)
    
    # Compute parcel profile
    parcel_profile = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0]).to('degC')
    
    # Compute CAPE and CIN
    cape, cin = mpcalc.surface_based_cape_cin(pressure, temperature, dewpoint)
    
    # Determine effective inflow layer based on CAPE and CIN thresholds
    effective_layer = (cape >= cape_threshold) & (cin >= cin_threshold)
    return effective_layer

# Example usage
date = datetime(2020, 5, 30, 12)
station = 'OUN'
df = WyomingUpperAir.request_data(date, station)

# Isolate needed variables and attach units
pressure = df['pressure'].values * units.hPa
temperature = df['temperature'].values * units.degC
dewpoint = df['dewpoint'].values * units.degC
u_wind, v_wind = mpcalc.wind_components(df['speed'].values * units.knots, df['direction'].values * units.degrees)

# Compute common sounding index parameters
lcl_pressure, lcl_temperature = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
lfc_pressure, lfc_temperature = mpcalc.lfc(pressure, temperature, dewpoint)
el_pressure, el_temperature = mpcalc.el(pressure, temperature, dewpoint)

# Compute parcel profile for a surface-based parcel
parcel_prof = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0])

# Compute LI, CAPE, CIN for a surface parcel
li = mpcalc.lifted_index(pressure, temperature, parcel_prof)
cape, cin = mpcalc.surface_based_cape_cin(pressure, temperature, dewpoint)

# Determine the LCL, LFC, and EL for the surface parcel
# Already computed above

# Compute characteristics of a mean layer parcel and the most unstable parcel
mean_layer_parcel = mpcalc.mixed_parcel(pressure, temperature, dewpoint, depth=100*units.hPa)
mu_parcel = mpcalc.most_unstable_parcel(pressure, temperature, dewpoint)

# Compute Bunkers Storm Motion vector
bunkers_motion = mpcalc.bunkers_storm_motion(pressure, temperature, dewpoint, u_wind, v_wind)

# Calculate the critical angle
critical_angle = mpcalc.critical_angle(pressure, temperature, dewpoint, u_wind, v_wind)

# Compute characteristics needed to compute the significant tornado parameter
stp = mpcalc.significant_tornado(pressure, temperature, dewpoint, u_wind, v_wind, lcl_height=lcl_pressure, mucape=mu_parcel[0], esrh=0)

# Compute the supercell composite parameter if possible
scp = mpcalc.supercell_composite(mu_parcel[0], mu_parcel[1], 0)

# Print out important sounding parameters
print(f"LCL Pressure: {lcl_pressure}, LCL Temperature: {lcl_temperature}")
print(f"LFC Pressure: {lfc_pressure}, LFC Temperature: {lfc_temperature}")
print(f"EL Pressure: {el_pressure}, EL Temperature: {el_temperature}")
print(f"LI: {li}, CAPE: {cape}, CIN: {cin}")
print(f"Mean Layer Parcel: {mean_layer_parcel}, Most Unstable Parcel: {mu_parcel}")
print(f"Bunkers Storm Motion: {bunkers_motion}, Critical Angle: {critical_angle}")
print(f"Significant Tornado Parameter: {stp}, Supercell Composite: {scp}")
```