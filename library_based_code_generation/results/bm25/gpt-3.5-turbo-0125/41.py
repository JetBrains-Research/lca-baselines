import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT
from metpy.io import get_upper_air_data

data = get_upper_air_data('obs', datetime(2022, 1, 1, 0), 33.9425, -118.4081)
data = data.dropna(subset=['temperature', 'dewpoint', 'pressure'])
data['temperature'] = data['temperature'].values * units.degC
data['dewpoint'] = data['dewpoint'].values * units.degC
data['pressure'] = data['pressure'].values * units.hPa

fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig)

skew.plot(data['pressure'], data['temperature'], 'r')
skew.plot(data['pressure'], data['dewpoint'], 'g')

lcl_pressure, lcl_temperature = mpcalc.lcl(data['pressure'][0], data['temperature'][0], data['dewpoint'][0])
skew.ax.plot(lcl_pressure, lcl_temperature, 'ko', markersize=5)

prof = mpcalc.parcel_profile(data['pressure'], data['temperature'][0], data['dewpoint'][0])
skew.plot(data['pressure'], prof, 'k')

cape, cin = mpcalc.cape_cin(data['pressure'], data['temperature'], data['dewpoint'], prof)
skew.shade_cin(data['pressure'], data['temperature'], prof)
skew.shade_cape(data['pressure'], data['temperature'], prof)

skew.ax.set_xlabel('Temperature (C)')
skew.ax.set_ylabel('Pressure (hPa)')

plt.show()