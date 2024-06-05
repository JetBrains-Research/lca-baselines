 ```python
import metpy.calc as mpcalc
from metpy.units import units
import metpy.plots as plots
import xarray as xr

def effective_inflow_layer(sounding):
    cin = -250 * units.Joules / units.kilogram
    cape = 100 * units.Joules / units.kilogram
    return sounding.where((sounding.potential_temperature - sounding.potential_temperature[0])*units.K >= cape, drop=True)\
                  .where((sounding.potential_temperature[0] - sounding.potential_temperature)*units.K >= cin, drop=True)

sounding = xr.open_dataset('sample_sounding.nc')

theta = sounding.potential_temperature.to(units.K)
theta_e = sounding.equivalent_potential_temperature.to(units.K)
u = sounding.wind_u.to(units.m/units.s)
v = sounding.wind_v.to(units.m/units.s)

wind_components = mpcalc.wind_components(u, v)

lcl = mpcalc.lcl(sounding.pressure, sounding.temperature, sounding.height)
lfc = mpcalc.lfc(theta, theta_e, sounding.pressure)
el = mpcalc.el(theta, theta_e, sounding.pressure)

parcel_profile = mpcalc.parcel_profile(sounding.pressure, sounding.temperature, sounding.height)
li, cape, cin = mpcalc.lifted_index(parcel_profile.pressure, parcel_profile.temperature, theta), \
                mpcalc.cape(parcel_profile.pressure, parcel_profile.temperature, theta), \
                mpcalc.cin(parcel_profile.pressure, parcel_profile.temperature, theta)

mean_layer_parcel = mpcalc.mean_layer_parcel(sounding.pressure, sounding.temperature, sounding.height)
most_unstable_parcel = mpcalc.most_unstable_parcel(sounding.pressure, sounding.temperature, theta)

bunkers_storm_motion = mpcalc.bunkers_storm_motion(wind_components.u, wind_components.v, sounding.height)
critical_angle = mpcalc.critical_angle(bunkers_storm_motion.u, bunkers_storm_motion.v, sounding.height)

sig_tornado_params = {'lcl': lcl, 'lfc': lfc, 'el': el, '0-3 km shear': mpcalc.shear(wind_components.u, wind_components.v, sounding.height, 0, 3000)*units.m/units.s,
                       '0-1 km helicity': mpcalc.helicity(sounding.height, wind_components.u, wind_components.v, most_unstable_parcel.theta_e)*units.m**2/units.s**2,
                       '0-3 km helicity': mpcalc.helicity(sounding.height, wind_components.u, wind_components.v, mean_layer_parcel.theta_e)*units.m**2/units.s**2,
                       'bunkers_storm_motion': bunkers_storm_motion, 'critical_angle': critical_angle}

supercell_comp_param = mpcalc.supercell_composite_parameter(sounding.pressure, sounding.temperature, sounding.height,
                                                            wind_components.u, wind_components.v, theta_e)

print("Important sounding parameters:")
print(f"LCL: {lcl:.2f} m")
print(f"LFC: {lfc:.2f} m")
print(f"EL: {el:.2f} m")
print(f"CAPE: {cape:.2f} J/kg")
print(f"CIN: {cin:.2f} J/kg")
print(f"LI: {li:.2f} K")
print(f"Mean layer parcel: {mean_layer_parcel.pressure.values[-1]:.2f} hPa, {mean_layer_parcel.temperature.values[-1]:.2f} K")
print(f"Most unstable parcel: {most_unstable_parcel.pressure.values[-1]:.2f} hPa, {most_unstable_parcel.temperature.values[-1]:.2f} K")
print(f"Supercell composite parameter: {supercell_comp_param:.2f}")
print(f"Significant tornado parameters: {sig_tornado_params}")
```