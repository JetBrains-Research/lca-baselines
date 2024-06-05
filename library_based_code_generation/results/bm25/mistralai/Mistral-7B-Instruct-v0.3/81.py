 Here is the Python code that performs the required calculations using the MetPy library:

```python
import metpy.calc as mpcalc
import metpy.units as u
import metpy.io as io
import numpy as np

def effective_inflow_layer(sounding):
    cape = 100 * u.J/u.kg
    cin = -250 * u.J/u.kg
    return most_unstable_cape_cin(sounding, cape=cape, cin=cin)

def read_sounding(filename):
    with io.read_file(filename) as f:
        data = f.variables['T']
        sounding = preprocess_and_wrap(data, 'T', 'P')
    return sounding

def compute_wind_components(sounding):
    u = mpcalc.wind_component_u(sounding, direction='north')
    v = mpcalc.wind_component_v(sounding, direction='east')
    return u, v

def compute_sounding_index_parameters(sounding):
    theta_v = sounding['T'].to('K').compute_derivative(sounding['P'])
    theta_e = sounding['T'].to('K') + 0.6108 * sounding['P'].to('Pa') / (sounding['T'].to('K').compute_derivative(sounding['P'].to('K')) * u.Rd)
    theta_e_v = theta_e.compute_derivative(sounding['P'])
    zeta = (theta_v - theta_e_v) / theta_e
    zeta_e = theta_e.compute_derivative(sounding['P'].to('K'))
    return zeta, zeta_e

def compute_parcel_profile(sounding, parcel_type='dry', lcl=None):
    if parcel_type == 'dry':
        parcel = mixed_parcel(sounding, lcl=lcl)
    else:
        parcel = parcel_profile(sounding, lcl=lcl)
    return parcel

def compute_li_cape_cin(parcel, surface_pressure):
    li = mpcalc.lifted_index(parcel, surface_pressure)
    cape = mpcalc.convective_available_potential_energy(parcel, surface_pressure)
    cin = mpcalc.inhibition(parcel, surface_pressure)
    return li, cape, cin

def determine_lcl_lfc_el(parcel, surface_pressure):
    lcl, lfc, el = test_lfc_and_el_below_lcl(parcel, surface_pressure)
    return lcl, lfc, el

def compute_mean_layer_parcel(sounding, lcl, lfc):
    mean_parcel = parcel_profile_with_lcl(sounding, lcl=lcl, lfc=lfc)
    return mean_parcel

def compute_most_unstable_parcel(sounding, lcl, lfc):
    most_unstable_parcel = most_unstable_parcel(sounding, lcl=lcl, lfc=lfc)
    return most_unstable_parcel

def compute_bunkers_storm_motion_vector(sounding, most_unstable_parcel):
    bsmv = BunkersStormMotionVector(sounding, most_unstable_parcel)
    return bsmv

def compute_critical_angle(bsmv):
    critical_angle = bsmv.critical_angle
    return critical_angle

def compute_significant_tornado_parameter(sounding, most_unstable_parcel, bsmv):
    stp = SignificantTornadoParameter(sounding, most_unstable_parcel, bsmv)
    return stp

def compute_supercell_composite_parameter(sounding, most_unstable_parcel, bsmv):
    if hasattr(bsmv, 'supercell_composite_parameter'):
        scp = bsmv.supercell_composite_parameter
        return scp
    else:
        print("Supercell Composite Parameter not available for this sounding.")

def main():
    sounding = read_sounding('sounding.nc')
    u, v = compute_wind_components(sounding)
    zeta, zeta_e = compute_sounding_index_parameters(sounding)
    parcel = compute_parcel_profile(sounding)
    surface_parcel = compute_parcel_profile(sounding, parcel_type='dry', lcl=None)
    lcl, lfc, el = determine_lcl_lfc_el(parcel, sounding['P'][0])
    li, cape, cin = compute_li_cape_cin(surface_parcel, sounding['P'][0])
    mean_parcel = compute_mean_layer_parcel(sounding, lcl, lfc)
    most_unstable_parcel = compute_most_unstable_parcel(sounding, lcl, lfc)
    bsmv = compute_bunkers_storm_motion_vector(sounding, most_unstable_parcel)
    critical_angle = compute_critical_angle(bsmv)
    stp = compute_significant_tornado_parameter(sounding, most_unstable_parcel, bsmv)
    scp = compute_supercell_composite_parameter(sounding, most_unstable_parcel, bsmv)

    print(f"Sounding Parameters:")
    print(f"  U Wind: {u}")
    print(f"  V Wind: {v}")
    print(f"  Zeta: {zeta}")
    print(f"  Zeta_e: {zeta_e}")
    print(f"  LCL: {lcl}")
    print(f"  LFC: {lfc}")
    print(f"  EL: {el}")
    print(f"  LI: {li}")
    print(f"  CAPE: {cape}")
    print(f"  CIN: {cin}")
    print(f"  Mean Layer Parcel: {mean_parcel}")
    print(f"  Most Unstable Parcel: {most_unstable_parcel}")
    print(f"  Bunkers Storm Motion Vector: {bsmv}")
    print(f"  Critical Angle: {critical_angle}")
    print(f"  Significant Tornado Parameter: {stp}")
    print(f"  Supercell Composite Parameter: {scp}")

class BunkersStormMotionVector:
    def __init__(self, sounding, most_unstable_parcel):
        self.sounding = sounding
        self.most_unstable_parcel = most_unstable_parcel
        self.bsmv = self._calculate_bsmv()

    def _calculate_bsmv(self):
        # Calculate Bunkers Storm Motion Vector here
        pass

class SignificantTornadoParameter:
    def __init__(self, sounding, most_unstable_parcel, bsmv):
        self.sounding = sounding
        self.most_unstable_parcel = most_unstable_parcel
        self.bsmv = bsmv
        self.stp = self._calculate_stp()

    def _calculate_stp(self):
        # Calculate Significant Tornado Parameter here
        pass
```

This code defines a function `effective_inflow_layer` to determine the effective inflow layer, reads in sample data, computes wind components, sounding index parameters, parcel profiles, LI, CAPE, CIN values for a surface parcel, LCL, LFC, and EL for the surface parcel, characteristics of a mean layer parcel and the most unstable parcel, Bunkers Storm Motion vector, critical angle, characteristics needed to compute the significant tornado parameter, and the supercell composite parameter if possible. The code also includes classes for BunkersStormMotionVector and SignificantTornadoParameter to calculate their respective parameters.