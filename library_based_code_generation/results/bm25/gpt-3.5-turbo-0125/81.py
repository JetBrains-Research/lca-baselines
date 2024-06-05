```python
import metpy.calc as mpcalc
from metpy.units import units
from metpy.testing import get_test_data

def calculate_sounding_parameters(data):
    p = data['pressure'].values * units.hPa
    T = data['temperature'].values * units.degC
    Td = data['dewpoint'].values * units.degC
    u = data['u_wind'].values * units.knot
    v = data['v_wind'].values * units.knot

    u, v = mpcalc.wind_components(u, v)
    profile = mpcalc.parcel_profile(p, T[0], Td[0])

    cape, cin = mpcalc.surface_based_cape_cin(p, T, Td, profile)
    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    lfc_pressure, lfc_temperature = mpcalc.lfc(p, T, Td, profile)
    el_pressure, el_temperature = mpcalc.el(p, T, Td, profile)

    mean_layer_parcel = mpcalc.mixed_parcel(p, T, Td)
    mu_parcel = mpcalc.most_unstable_parcel(p, T, Td)

    bunkers_storm_motion = mpcalc.bunkers_storm_motion(p, u, v)
    critical_angle = mpcalc.critical_angle(p, u, v, bunkers_storm_motion)

    sig_tor_param = mpcalc.significant_tornado_parameter(p, T, Td, u, v)
    supercell_comp_param = mpcalc.supercell_composite(p, T, Td, u, v)

    print(f"CAPE: {cape}, CIN: {cin}")
    print(f"LCL: {lcl_pressure}, LFC: {lfc_pressure}, EL: {el_pressure}")
    print(f"Mean Layer Parcel: {mean_layer_parcel}")
    print(f"Most Unstable Parcel: {mu_parcel}")
    print(f"Bunkers Storm Motion: {bunkers_storm_motion}, Critical Angle: {critical_angle}")
    print(f"Significant Tornado Parameter: {sig_tor_param}")
    print(f"Supercell Composite Parameter: {supercell_comp_param}")

data = get_test_data('sounding_data.nc')
calculate_sounding_parameters(data)
```