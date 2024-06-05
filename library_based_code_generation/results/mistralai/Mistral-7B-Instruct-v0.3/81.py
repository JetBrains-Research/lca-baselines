 Here is a Python code that performs the requested calculations using the MetPy library. Please note that you need to have MetPy installed and the necessary sounding data.

```python
import metpy.calc as mpcalc
import metpy.units as u
import numpy as np
from metpy.metdata.parsers import Stations

def effective_inflow_layer(sounding):
    cape = mpcalc.convective_available_potential_energy(sounding, units=u.J/u.kg)
    cin = mpcalc.inflow_layer_stability(sounding, units=u.J/u.kg)
    return cape > 100 * u.J/u.kg and cin < -250 * u.J/u.kg

def read_sounding(filename):
    with Stations(filename) as stn:
        sounding = stn[0].sounding
    return sounding

def compute_wind_components(sounding):
    u = sounding['u'].m
    v = sounding['v'].m
    w = sounding['w'].m
    return u, v, w

def compute_sounding_index_parameters(sounding):
    theta = sounding['potential_temperature'].K
    p = sounding['pressure'].hPa
    t = sounding['temperature'].K
    return mpcalc.lifted_index(theta, t, p), mpcalc.equivalent_potential_temperature(theta, p), mpcalc.static_stability(t, p)

def compute_parcel_profile(sounding, parcel_temperature, parcel_humidity):
    parcel_theta = mpcalc.potential_temperature(parcel_temperature, parcel_humidity, sounding['pressure'])
    return sounding['pressure'][::-1], parcel_theta[::-1]

def compute_li_cape_cin(sounding, parcel_temperature, parcel_humidity):
    li, theta_e, sigma_t = compute_sounding_index_parameters(sounding)
    lcl = mpcalc.lifting_condensation_level(parcel_temperature, parcel_humidity, sounding['pressure'])
    lcl_theta = mpcalc.potential_temperature(parcel_temperature[lcl], parcel_humidity[lcl], sounding['pressure'][lcl])
    cape = mpcalc.convective_available_potential_energy(sounding, parcel_theta, parcel_humidity, lcl)
    cin = mpcalc.inflow_layer_stability(sounding, parcel_theta, parcel_humidity, lcl)
    return li.to('K'), lcl_theta.to('K'), cape.to(u.J/u.kg), cin.to(u.J/u.kg)

def compute_lcl_lfc_el(sounding, parcel_temperature, parcel_humidity):
    li, lcl_theta, cape, cin = compute_li_cape_cin(sounding, parcel_temperature, parcel_humidity)
    lfc = mpcalc.level_of_free_convection(cape, cin, lcl_theta)
    el = mpcalc.elevation(sounding['pressure'], 1000 * u.hPa)
    return lcl, lfc, el

def compute_mean_layer_parcel(sounding, lcl, lfc):
    lcl_index = sounding['pressure'].index(lcl)
    lfc_index = sounding['pressure'].index(lfc)
    mean_layer_parcel_theta = np.mean(sounding['potential_temperature'][lcl_index:lfc_index])
    return mean_layer_parcel_theta

def compute_most_unstable_parcel(sounding, lcl, lfc):
    lcl_index = sounding['pressure'].index(lcl)
    lfc_index = sounding['pressure'].index(lfc)
    most_unstable_parcel_theta = max(sounding['potential_temperature'][lcl_index:lfc_index])
    return most_unstable_parcel_theta

def compute_bunkers_storm_motion_vector(sounding, lcl, lfc, most_unstable_parcel_theta):
    lcl_index = sounding['pressure'].index(lcl)
    lfc_index = sounding['pressure'].index(lfc)
    u_star = (most_unstable_parcel_theta - sounding['potential_temperature'][lcl_index]) / (lfc - lcl)
    return u_star

def compute_critical_angle(bunkers_storm_motion_vector):
    return np.arctan(bunkers_storm_motion_vector)

def compute_significant_tornado_parameter(sounding, lcl, lfc, most_unstable_parcel_theta, bunkers_storm_motion_vector):
    sigma_c = mpcalc.static_stability(sounding['temperature'][lcl], sounding['pressure'][lcl])
    sigma_e = mpcalc.equivalent_potential_temperature(most_unstable_parcel_theta, sounding['pressure'][lcl])
    return (sigma_c - sigma_e) * bunkers_storm_motion_vector

def compute_supercell_composite_parameter(sounding, lcl, lfc, most_unstable_parcel_theta, bunkers_storm_motion_vector):
    if bunkers_storm_motion_vector > 0.01:
        return compute_significant_tornado_parameter(sounding, lcl, lfc, most_unstable_parcel_theta, bunkers_storm_motion_vector)
    else:
        return np.nan

def main():
    filename = 'sounding.txt'
    sounding = read_sounding(filename)
    u, v, w = compute_wind_components(sounding)
    parcel_temperature = sounding['temperature'].m
    parcel_humidity = sounding['specific_humidity'].m

    if effective_inflow_layer(sounding):
        li, lcl_theta, cape, cin = compute_li_cape_cin(sounding, parcel_temperature, parcel_humidity)
        lcl, lfc, el = compute_lcl_lfc_el(sounding, parcel_temperature, parcel_humidity)
        mean_layer_parcel_theta = compute_mean_layer_parcel(sounding, lcl, lfc)
        most_unstable_parcel_theta = compute_most_unstable_parcel(sounding, lcl, lfc)
        bunkers_storm_motion_vector = compute_bunkers_storm_motion_vector(sounding, lcl, lfc, most_unstable_parcel_theta)
        critical_angle = compute_critical_angle(bunkers_storm_motion_vector)
        significant_tornado_parameter = compute_significant_tornado_parameter(sounding, lcl, lfc, most_unstable_parcel_theta, bunkers_storm_motion_vector)
        supercell_composite_parameter = compute_supercell_composite_parameter(sounding, lcl, lfc, most_unstable_parcel_theta, bunkers_storm_motion_vector)

        print(f"LI: {li}")
        print(f"LCL: {lcl} hPa, {lcl_theta} K")
        print(f"LFC: {lfc} hPa, {lfc_theta} K")
        print(f"EL: {el} m")
        print(f"Mean Layer Parcel Potential Temperature: {mean_layer_parcel_theta} K")
        print(f"Most Unstable Parcel Potential Temperature: {most_unstable_parcel_theta} K")
        print(f"Bunkers Storm Motion Vector: {bunkers_storm_motion_vector} s-1")
        print(f"Critical Angle: {critical_angle} rad")
        print(f"Significant Tornado Parameter: {significant_tornado_parameter} J/kg m")
