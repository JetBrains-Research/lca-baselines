import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT
from metpy.calc import get_layer
from metpy.calc import lcl, lfc, el
from metpy.calc import bunkers_storm_motion
from metpy.calc import significant_tornado_parameter
from metpy.calc import supercell_composite
from metpy.calc import most_unstable_cape_cin
from metpy.calc import bulk_shear
from metpy.calc import mean_layer_parcel
from metpy.calc import effective_inflow_layer

def calculate_sounding_parameters(data):
    p = data['pressure'].values * units.hPa
    T = data['temperature'].values * units.degC
    Td = data['dewpoint'].values * units.degC
    u = data['u_wind'].values * units.knot
    v = data['v_wind'].values * units.knot

    # Compute wind components
    wind_speed = np.sqrt(u**2 + v**2)
    wind_dir = mpcalc.wind_direction(u, v)

    # Compute common sounding index parameters
    cape, cin = most_unstable_cape_cin(p, T, Td, u, v)

    # Compute parcel profile for surface-based parcel
    surface_parcel = mpcalc.parcel_profile(p, T[0], Td[0])

    # Compute LI, CAPE, CIN for surface parcel
    surface_li = mpcalc.lapse_rate(p, T, Td, 0)
    surface_lcl, surface_lfc, surface_el = lcl(p, T, Td)
    surface_cape, surface_cin = mpcalc.cape_cin(p, T, Td, surface_parcel)

    # Compute characteristics of mean layer parcel
    mean_layer = mean_layer_parcel(p, T, Td)

    # Compute most unstable parcel
    mu_parcel = mpcalc.most_unstable_parcel(p, T, Td)

    # Compute Bunkers Storm Motion vector
    bunkers_motion = bunkers_storm_motion(p, u, v, p[0])

    # Calculate critical angle
    critical_angle = mpcalc.critical_angle(p, u, v, bunkers_motion)

    # Compute significant tornado parameter
    sig_tor_param = significant_tornado_parameter(p, T, Td, u, v)

    # Compute supercell composite parameter
    supercell_comp = supercell_composite(p, T, Td, u, v)

    # Print out important sounding parameters
    print(f'CAPE: {cape}')
    print(f'CIN: {cin}')
    print(f'LCL: {surface_lcl}')
    print(f'LFC: {surface_lfc}')
    print(f'EL: {surface_el}')
    print(f'Mean Layer Parcel: {mean_layer}')
    print(f'Most Unstable Parcel: {mu_parcel}')
    print(f'Bunkers Storm Motion: {bunkers_motion}')
    print(f'Critical Angle: {critical_angle}')
    print(f'Significant Tornado Parameter: {sig_tor_param}')
    print(f'Supercell Composite Parameter: {supercell_comp}')

# Read in sample data
data = pd.read_csv('sample_data.csv')

# Isolate needed variables and attach units
data['pressure'] = data['pressure'].values * units.hPa
data['temperature'] = data['temperature'].values * units.degC
data['dewpoint'] = data['dewpoint'].values * units.degC
data['u_wind'] = data['u_wind'].values * units.knot
data['v_wind'] = data['v_wind'].values * units.knot

# Compute effective inflow layer
eff_inflow_layer = effective_inflow_layer(data['pressure'], data['temperature'], data['dewpoint'], cape=100 * units('J/kg'), cin=-250 * units('J/kg'))

# Call function to calculate sounding parameters
calculate_sounding_parameters(data)