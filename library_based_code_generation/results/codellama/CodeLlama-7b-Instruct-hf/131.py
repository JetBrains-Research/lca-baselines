  import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# Create a series of maps using Basemap
maps = []
for i in range(6):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=Basemap(projection='ortho',
                                                 lat_0=45, lon_0=-100,
                                                 resolution='l',
                                                 area_thresh=10000))
    maps.append(ax)

# Add coastlines, country boundaries, and continents
for map in maps:
    map.drawcoastlines()
    map.drawcountries()
    map.fillcontinents(color='#dddddd')

# Add meridians and parallels
for map in maps:
    map.drawmeridians(np.arange(0, 360, 30))
    map.drawparallels(np.arange(-90, 90, 30))

# Add cities and their names
cities = [('New York', 40.7128, -74.0060),
          ('Los Angeles', 34.0522, -118.2437),
          ('Chicago', 41.8781, -87.6298),
          ('Houston', 29.7633, -95.3633),
          ('Phoenix', 33.4484, -112.0740)]
for map in maps:
    for city, lat, lon in cities:
        x, y = map(lon, lat)
        map.plot(x, y, 'o', markersize=10, markerfacecolor='#ff0000', markeredgecolor='#000000', markeredgewidth=1)
        map.text(x, y, city, fontsize=10, fontweight='bold')

# Generate data on a regular lat/lon grid
lats = np.linspace(0, 90, 100)
lons = np.linspace(0, 360, 100)
data = np.sin(np.deg2rad(lats)) * np.cos(np.deg2rad(lons))

# Contour the data over the map
for map in maps:
    cs = map.contourf(lons, lats, data, 15, cmap='RdYlGn')

# Add backgrounds
for map in maps:
    map.background_img(map.etopo, zorder=0)
    map.background_img(map.land, zorder=1)
    map.background_img(map.ocean, zorder=2)
    map.background_img(map.shadedrelief, zorder=3)
    map.background_img(map.bluemarble, zorder=4)
    map.background_img(map.filledcontinents, zorder=5)

# Display all the generated maps
plt.show()