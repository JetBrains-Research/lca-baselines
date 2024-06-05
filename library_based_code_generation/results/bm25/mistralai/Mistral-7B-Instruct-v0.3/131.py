 Here is a simplified version of the code that should help you get started. Note that you need to have Basemap and Matplotlib installed. Also, the city coordinates and data generation are placeholders and might need adjustments.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import geopandas as gpd

# Load shapefile with city coordinates
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
cities = cities[cities.pop_max <= 1000000]  # Filter cities by population

# Define the projection and create the Basemap object
lat0, lon0 = 45, -100
resolution = 'l'
projection = 'ortho'
llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = -180, -90, 180, 90

m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
            resolution=resolution, projection=projection, lat_ts=lat0, lon_ts=lon0)

# Generate data on a regular lat/lon grid
x, y = m(np.linspace(llcrnrlon, urcrnrlon, 100), np.linspace(llcrnrlat, urcrnrlat, 100))
z = np.sin(x)*np.cos(y)

# Create a contour plot of the data
cs = m.contour(x, y, z, colors='k')

# Plot coastlines, country boundaries, and continents
m.drawcoastlines()
m.drawcountries()
m.fillcontinents()

# Draw meridians and parallels every 30 degrees
m.drawmeridians(np.arange(llcrnrlon, urcrnrlon + 360, 30), labels=[1,0,0,0])
m.drawparallels(np.arange(llcrnrlat, urcrnrlat + 90, 30), labels=[0,0,0,1])

# Plot cities
for city in cities.head(5):
    x, y = m(city.longitude, city.latitude)
    m.plot(x, y, 'ko', markersize=10)
    plt.text(x, y, city.name, fontsize=12)

# Display the map with different backgrounds
for background in ['continent', 'ocean', 'blue_marble', 'shaded_relief', 'etopo', 'etopo_with_transparent_land']:
    m.bluemarble(scale=0.5, shade_relief=True)
    if background == 'continent':
        m.fillcontinents(color='white')
    elif background == 'ocean':
        m.fillcontinents(color='white', lake_color='white')
    elif background == 'blue_marble':
        m.drawmapboundary(fill_color='white')
    elif background == 'shaded_relief':
        pass
    elif background == 'etopo':
        m.etopo(scale=0.5)
    elif background == 'etopo_with_transparent_land':
        m.etopo(scale=0.5, alpha=0.5)
    plt.show()
```

This code creates a series of maps with different backgrounds and plots the locations of five cities. However, it does not include the data generation part for contouring the data, as it is quite complex and beyond the scope of this response. You can find examples of data generation in the Basemap documentation or other resources.