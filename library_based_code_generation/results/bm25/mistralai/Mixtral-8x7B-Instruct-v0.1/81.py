 from metpy.calc import (
parcel_profile, mixed_parcel, most_unstable_parcel,
most_unstable_cape_cin, check_and_drop_units,
)
from metpy.plots import SkewT
from metpy.units import units
from metpy.sounding import Sounding
import metpy.interpolate as interpolate
import numpy as np

def effective\_inflow\_layer(sounding\_data):
sounding = Sounding(sounding\_data, units='mb', height\_unit='km')
parcel = parcel\_profile(sounding.pressure, sounding.temperature,
sounding.specific\_humidity, sounding.height)
mu\_parcel = most\_unstable\_parcel(sounding.pressure, sounding.temperature,
sounding.specific\_humidity)
mu\_cape, mu\_cin = most\_unstable\_cape\_cin(sounding.pressure,
sounding.temperature, sounding.specific\_humidity,
parcel.pressure, parcel.temperature, parcel.specific\_humidity)
lcl\_pressure, lcl\_height = test\_parcel\_profile\_lcl(sounding.pressure,
sounding.temperature, sounding.specific\_humidity,
mu\_parcel.pressure, mu\_parcel.temperature,
mu\_parcel.specific\_humidity)
lfc\_pressure, lfc\_height = test\_lfc\_and\_el\_below\_lcl(
sounding.pressure, sounding.temperature,
sounding.specific\_humidity, mu\_parcel.pressure,
mu\_parcel.temperature, mu\_parcel.specific\_humidity,
lcl\_pressure)
el\_pressure, el\_height = test\_lfc\_and\_el\_below\_lcl(
sounding.pressure, sounding.temperature,
sounding.specific\_humidity, mu\_parcel.pressure,
mu\_parcel.temperature, mu\_parcel.specific\_humidity,
lfc\_pressure, return\_lcl=False)
mu\_sounding = mixed\_parcel(sounding.pressure, sounding.temperature,
sounding.specific\_humidity, lcl\_pressure,
mu\_parcel.temperature, mu\_parcel.specific\_humidity)
return {'mu\_parcel': mu\_parcel,
'mu\_cape': mu\_cape,
'mu\_cin': mu\_cin,
'lcl\_pressure': lcl\_pressure,
'lfc\_pressure': lfc\_pressure,
'el\_pressure': el\_pressure,
'mu\_sounding': mu\_sounding}

sounding\_data = return\_the\_kwargs()
result = effective\_inflow\_layer(sounding\_data)
print(f'Most unstable parcel CAPE: {result["mu\_cape"]} J/kg')
print(f'Most unstable parcel CIN: {result["mu\_cin"]} J/kg')
print(f'LCL pressure: {result["lcl\_pressure"]} mb')
print(f'LFC pressure: {result["lfc\_pressure"]} mb')
print(f'EL pressure: {result["el\_pressure"]} mb')