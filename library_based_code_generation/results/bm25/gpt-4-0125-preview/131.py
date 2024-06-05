```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Define the cities with their coordinates and names
cities = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740}
]

# Generate data on a regular lat/lon grid
nlats = 73; nlons = 145; delta = 2.*np.pi/(nlons-1)
lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
lons = (delta*np.indices((nlats,nlons))[1,:,:])
wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)

data = wave + mean

# Background types for demonstration
backgrounds = ['fillcontinents', 'bluemarble', 'shadedrelief', 'etopo']

fig = plt.figure(figsize=(14, 10))

for i, background in enumerate(backgrounds, start=1):
    ax = fig.add_subplot(2, 2, i)
    m = Basemap(projection='ortho', lat_0=45, lon_0=-100, resolution='l')
    
    # Draw coastlines, country boundaries, fill continents.
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90., 120., 30.))
    m.drawmeridians(np.arange(0., 420., 30.))
    
    if background == 'fillcontinents':
        m.fillcontinents(color='coral', lake_color='aqua')
    elif background == 'bluemarble':
        m.bluemarble()
    elif background == 'shadedrelief':
        m.shadedrelief()
    elif background == 'etopo':
        m.etopo()
    
    # Plot cities
    for city in cities:
        x, y = m(city['lon'], city['lat'])
        m.plot(x, y, 'bo', markersize=5)
        plt.text(x, y, city['name'], fontsize=12, ha='right')
    
    # Contour data over the map
    x, y = np.meshgrid(np.linspace(-180, 180, nlons), np.linspace(-90, 90, nlats))
    x, y = m(x, y)
    m.contour(x, y, data, 15, linewidths=1.5, colors='k')

plt.show()
```