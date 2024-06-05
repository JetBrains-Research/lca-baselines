 ```python
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# Create a regular lat/lon grid
lats = np.linspace(20, 70, 100)
lons = np.linspace(-130, -60, 100)
lat, lon = np.meshgrid(lats, lons)
data = np.sin(lat/10)*np.cos(lon/10)

# Define the maps
projection = 'ortho'
central_lat = 45
central_lon = -100

fig, axs = plt.subplots(6, 1, figsize=(8, 12))

# Filled continent
map = Basemap(projection=projection, lon_0=central_lon, lat_0=central_lat,
              resolution='l', area_thresh=1000, ax=axs[0])
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='coral')
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))
map.plot(5, 45, 'ko', markersize=5)
map.text(6, 45, 'City 1', fontsize=8, ha='center', va='bottom')
map.contour(lon, lat, data, levels=np.linspace(-1, 1, 11), colors='black')

# Land-sea mask
map = Basemap(projection=projection, lon_0=central_lon, lat_0=central_lat,
              resolution='l', area_thresh=1000, ax=axs[1])
map.drawcoastlines()
map.drawcountries()
map.drawmapboundary(fill_color='aqua')
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))
map.plot(5, 45, 'ko', markersize=5)
map.text(6, 45, 'City 2', fontsize=8, ha='center', va='bottom')
map.contour(lon, lat, data, levels=np.linspace(-1, 1, 11), colors='black')

# Blue marble
map = Basemap(projection=projection, lon_0=central_lon, lat_0=central_lat,
              resolution='l', area_thresh=1000, ax=axs[2])
map.bluemarble()
map.drawcoastlines()
map.drawcountries()
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))
map.plot(5, 45, 'ko', markersize=5)
map.text(6, 45, 'City 3', fontsize=8, ha='center', va='bottom')
map.contour(lon, lat, data, levels=np.linspace(-1, 1, 11), colors='black')

# Shaded relief
map = Basemap(projection=projection, lon_0=central_lon, lat_0=central_lat,
              resolution='l', area_thresh=1000, ax=axs[3])
map.shadedrelief()
map.drawcoastlines()
map.drawcountries()
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))
map.plot(5, 45, 'ko', markersize=5)
map.text(6, 45, 'City 4', fontsize=8, ha='center', va='bottom')
map.contour(lon, lat, data, levels=np.linspace(-1, 1, 11), colors='black')

# Etopo
map = Basemap(projection=projection, lon_0=central_lon, lat_0=central_lat,
              resolution='l', area_thresh=1000, ax=axs[4])
map.etopo()
map.drawcoastlines()
map.drawcountries()
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))
map.plot(5, 45, 'ko', markersize=5)
map.text(6, 45, 'City 5', fontsize=8, ha='center', va='bottom')
map.contour(lon, lat, data, levels=np.linspace(-1, 1, 11), colors='black')

# Etopo with transparent land areas
map = Basemap(projection=projection, lon_0=central_lon, lat_0=central_lat,
              resolution='l', area_thresh=1000, ax=axs[5])
map.etopo(transparent=True)
map.drawcoastlines()
map.drawcountries()
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))
map.plot(5, 45, 'ko', markersize=5)
map.text(6, 45, 'City 6', fontsize=8, ha='center', va='bottom')
map.contour(lon, lat, data, levels=np.linspace(-1, 1, 11), colors='black')

plt.show()
```