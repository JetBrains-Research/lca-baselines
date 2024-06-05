```python
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Create Basemap instance
m = Basemap(projection='ortho', lat_0=45, lon_0=-100)

# Draw coastlines, country boundaries, and fill continents with color
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='coral', lake_color='aqua')

# Draw meridians and parallels every 30 degrees
m.drawmeridians(range(0, 360, 30))
m.drawparallels(range(-90, 90, 30))

# Plot cities with names displayed
cities = {
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Miami': (25.7617, -80.1918)
}
for city, (lat, lon) in cities.items():
    x, y = m(lon, lat)
    m.plot(x, y, 'bo')  # Filled circle
    plt.text(x, y, city, fontsize=8, ha='left')

# Generate data on a regular lat/lon grid and contour over the map
lons, lats = np.meshgrid(np.linspace(-180, 180, 100), np.linspace(-90, 90, 100))
data = np.sin(np.radians(lons)) * np.cos(np.radians(lats))
x, y = m(lons, lats)
m.contour(x, y, data, levels=10, colors='k')

# Display the map with different backgrounds
plt.figure()
m.bluemarble()
plt.figure()
m.shadedrelief()
plt.figure()
m.etopo()
plt.figure()
m.etopo()
m.drawlsmask(land_color='coral', ocean_color='aqua')

plt.show()
```