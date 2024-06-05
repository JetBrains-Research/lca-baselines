```python
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.io import get_upper_air_data
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir

def effective_inflow_layer(pressure, temperature, dewpoint, u_wind, v_wind, cape_threshold=100 * units('J/kg'), cin_threshold=-250 * units('J/kg')):
    """
    Determine the effective inflow layer based on CAPE and CIN thresholds.
    """
    # Calculate parcel profile
    prof = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0]).to('degC')
    
    # Calculate CAPE and CIN
    cape, cin = mpcalc.surface_based_cape_cin(pressure, temperature, dewpoint)
    
    # Identify effective inflow layer based on CAPE and CIN thresholds
    effective_layer_mask = (cape >= cape_threshold) & (cin >= cin_threshold)
    effective_pressure = pressure[effective_layer_mask]
    
    if effective_pressure.size > 0:
        bottom_pressure = effective_pressure[0]
        top_pressure = effective_pressure[-1]
    else:
        bottom_pressure = np.nan * units.hPa
        top_pressure = np.nan * units.hPa
    
    return bottom_pressure, top_pressure

# Sample data loading
date = datetime(2023, 5, 1, 0)
station = 'OUN'
df = WyomingUpperAir.request_data(date, station)

# Isolate needed variables and attach units
pressure = df['pressure'].values * units(df.units['pressure'])
temperature = df['temperature'].values * units(df.units['temperature'])
dewpoint = df['dewpoint'].values * units(df.units['dewpoint'])
u_wind, v_wind = mpcalc.wind_components(df['speed'].values * units(df.units['speed']),
                                         df['direction'].values * units.deg)

# Compute common sounding index parameters
lcl_pressure, lcl_temperature = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
lfc_pressure, lfc_temperature = mpcalc.lfc(pressure, temperature, dewpoint)
el_pressure, el_temperature = mpcalc.el(pressure, temperature, dewpoint)

# Compute the parcel profile for a surface-based parcel
parcel_profile = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0])

# Compute corresponding LI, CAPE, CIN values for a surface parcel
li = mpcalc.lifted_index(pressure, temperature, parcel_profile)
cape, cin = mpcalc.surface_based_cape_cin(pressure, temperature, dewpoint)

# Compute characteristics of a mean layer parcel and the most unstable parcel
mean_layer_parcel = mpcalc.mixed_layer(pressure, temperature, dewpoint, depth=100 * units.hPa)
most_unstable_parcel = mpcalc.most_unstable_parcel(pressure, temperature, dewpoint)

# Compute the Bunkers Storm Motion vector
storm_motion = mpcalc.bunkers_storm_motion(pressure, u_wind, v_wind, depth=6000 * units.meter)

# Calculate the critical angle
critical_angle = mpcalc.critical_angle(pressure, temperature, dewpoint, u_wind, v_wind)

# Compute characteristics needed to compute the significant tornado parameter
stp = mpcalc.significant_tornado(pressure, temperature, dewpoint, u_wind, v_wind, lcl_height=lcl_pressure, lfc_height=lfc_pressure, el_height=el_pressure)

# Compute the supercell composite parameter if possible
scp = mpcalc.supercell_composite(pressure, temperature, dewpoint, u_wind, v_wind, mucape=cape, effective_bulk_shear=storm_motion)

# Print out important sounding parameters
print(f"LCL Pressure: {lcl_pressure}, LCL Temperature: {lcl_temperature}")
print(f"LFC Pressure: {lfc_pressure}, LFC Temperature: {lfc_temperature}")
print(f"EL Pressure: {el_pressure}, EL Temperature: {el_temperature}")
print(f"LI: {li}, CAPE: {cape}, CIN: {cin}")
print(f"Mean Layer Parcel: {mean_layer_parcel}, Most Unstable Parcel: {most_unstable_parcel}")
print(f"Bunkers Storm Motion: {storm_motion}, Critical Angle: {critical_angle}")
print(f"Significant Tornado Parameter: {stp}, Supercell Composite: {scp}")
```