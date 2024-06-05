```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Define the cities with their coordinates and names
cities = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740)
}

# Generate some data on a regular lat/lon grid
nlats = 73; nlons = 145; delta = 2.*np.pi/(nlons-1)
lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
lons = (delta*np.indices((nlats,nlons))[1,:,:])
wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)

# List of backgrounds
backgrounds = ['fillcontinents', 'bluemarble', 'shadedrelief', 'etopo']

fig = plt.figure(figsize=(12, 10))

for i, background in enumerate(backgrounds, start=1):
    ax = fig.add_subplot(2, 2, i)
    m = Basemap(projection='ortho', lat_0=45, lon_0=-100, resolution='l')

    # Draw coastlines, country boundaries, fill continents.
    m.drawcoastlines()
    m.drawcountries()
    if background == 'fillcontinents':
        m.fillcontinents(color='coral', lake_color='aqua')
    elif background == 'bluemarble':
        m.bluemarble()
    elif background == 'shadedrelief':
        m.shadedrelief()
    elif background == 'etopo':
        m.etopo()

    # Draw meridians and parallels.
    m.drawmeridians(np.arange(-180, 180, 30))
    m.drawparallels(np.arange(-90, 90, 30))

    # Plot cities
    for city, (lat, lon) in cities.items():
        x, y = m(lon, lat)
        m.plot(x, y, 'bo', markersize=5)
        plt.text(x, y, city, fontsize=12)

    # Contour data over the map
    x, y = m(*np.meshgrid(lons*180./np.pi, lats*180./np.pi))
    m.contour(x, y, wave+mean, 15, linewidths=1.5)

plt.show()
```