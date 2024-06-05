  Here is the Python code based on the given instructions:
```
import numpy as np
import xarray as xr
from metpy.calc import *
from metpy.units import units

# Define function to determine effective inflow layer for convective sounding
def effective_inflow_layer(cape, cin, p, T, z):
    # Compute wind components
    u, v = wind_components(p, T, z)

    # Compute common sounding index parameters
    theta_e = theta_e_from_T(T)
    theta_v = theta_v_from_T(T)
    theta_l = theta_l_from_T(T)

    # Compute parcel profile for surface-based parcel
    parcel = parcel_profile(p, T, z, theta_e, theta_v, theta_l)

    # Compute corresponding LI, CAPE, CIN values for surface parcel
    li, cape, cin = parcel_profile_with_lcl(p, T, z, theta_e, theta_v, theta_l)

    # Determine LCL, LFC, and EL for surface parcel
    lcl, lfc, el = lcl_lfc_el(p, T, z, theta_e, theta_v, theta_l)

    # Compute characteristics of mean layer parcel and most unstable parcel
    mean_layer_parcel = parcel_profile(p, T, z, theta_e, theta_v, theta_l)
    most_unstable_parcel = most_unstable_parcel(p, T, z, theta_e, theta_v, theta_l)

    # Compute Bunkers Storm Motion vector and use it to calculate critical angle
    bunkers_storm_motion = bunkers_storm_motion(p, T, z, theta_e, theta_v, theta_l)
    critical_angle = critical_angle(bunkers_storm_motion)

    # Compute characteristics needed to compute significant tornado parameter
    significant_tornado_parameter = significant_tornado_parameter(p, T, z, theta_e, theta_v, theta_l)

    # Compute supercell composite parameter if possible
    supercell_composite_parameter = supercell_composite_parameter(p, T, z, theta_e, theta_v, theta_l)

    # Print out important sounding parameters
    print("Effective Inflow Layer:", effective_inflow_layer)
    print("LCL:", lcl)
    print("LFC:", lfc)
    print("EL:", el)
    print("LI:", li)
    print("CAPE:", cape)
    print("CIN:", cin)
    print("Bunkers Storm Motion:", bunkers_storm_motion)
    print("Critical Angle:", critical_angle)
    print("Significant Tornado Parameter:", significant_tornado_parameter)
    print("Supercell Composite Parameter:", supercell_composite_parameter)

# Read in sample data
ds = xr.open_dataset("sounding_data.nc")

# Isolate needed variables and attach units
p = ds.pressure.to(units.Pa)
T = ds.temperature.to(units.degC)
z = ds.height.to(units.m)

# Compute effective inflow layer for convective sounding
effective_inflow_layer(cape=100, cin=-250, p=p, T=T, z=z)
```
Note that this code assumes that the sounding data is stored in a NetCDF file called "sounding_data.nc" and that the variables "pressure", "temperature", and "height" are available in the file. The code also assumes that the units of the variables are in Pa, degC, and m, respectively. If the units are different, you will need to modify the code accordingly.