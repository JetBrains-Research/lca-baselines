  import numpy as np
import pandas as pd
from metpy.units import units
from metpy.calc import thermo

def effective_inflow_layer(sounding):
    # Define default values for CAPE and CIN
    CAPE = 100 * units.J / units.kg
    CIN = -250 * units.J / units.kg

    # Isolate needed variables and attach units
    temperature = sounding['temperature'] * units.degC
    dewpoint = sounding['dewpoint'] * units.degC
    pressure = sounding['pressure'] * units.hPa
    wind_speed = sounding['wind_speed'] * units.m/units.s
    wind_direction = sounding['wind_direction'] * units.deg

    # Compute wind components
    u_wind = wind_speed * np.cos(wind_direction)
    v_wind = wind_speed * np.sin(wind_direction)

    # Compute common sounding index parameters
    theta_e = thermo.theta_e(temperature, dewpoint)
    theta_e_s = thermo.theta_e(temperature, dewpoint, pressure)
    lcl_pressure = thermo.lcl(theta_e, theta_e_s)

    # Compute the parcel profile for a surface-based parcel
    parcel_profile = thermo.parcel_profile(pressure, temperature, dewpoint)

    # Compute corresponding LI, CAPE, CIN values for a surface parcel
    li = thermo.lifted_index(parcel_profile)
   cape = thermo.cape(parcel_profile)
    cin = thermo.cin(parcel_profile)

    # Determine the LCL, LFC, and EL for the surface parcel
    lcl = thermo.lcl(temperature, dewpoint)
    lfc = thermo.lfc(temperature, dewpoint)
    el = thermo.el(temperature, dewpoint)

    # Compute characteristics of a mean layer parcel and the most unstable parcel
    mean_layer_parcel = thermo.mean_layer_parcel(pressure, temperature, dewpoint)
    most_unstable_parcel = thermo.most_unstable_parcel(pressure, temperature, dewpoint)

    # Compute the Bunkers Storm Motion vector and use it to calculate the critical angle
    storm_motion = thermo.storm_motion(u_wind, v_wind)
    critical_angle = thermo.critical_angle(storm_motion)

    # Compute the characteristics needed to compute the significant tornado parameter
    significant_tornado_parameter = thermo.significant_tornado_parameter(cape, cin)

    # Compute the supercell composite parameter if possible
    supercell_composite_parameter = thermo.supercell_composite_parameter(cape, cin)

    # Print out the important sounding parameters
    print("Effective Inflow Layer:", effective_inflow_layer)
    print("LCL Pressure:", lcl_pressure)
    print("LCL Temperature:", lcl.temperature)
    print("LCL Dewpoint:", lcl.dewpoint)
    print("LFC Pressure:", lfc.pressure)
    print("LFC Temperature:", lfc.temperature)
    print("LFC Dewpoint:", lfc.dewpoint)
    print("EL Pressure:", el.pressure)
    print("EL Temperature:", el.temperature)
    print("EL Dewpoint:", el.dewpoint)
    print("LI:", li)
    print("CAPE:", cape)
    print("CIN:", cin)
    print("Significant Tornado Parameter:", significant_tornado_parameter)
    print("Supercell Composite Parameter:", supercell_composite_parameter)
    print("Critical Angle:", critical_angle)