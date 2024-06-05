```python
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

cities = {
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Phoenix': (33.4484, -112.0740)
}

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for ax, background in zip(axs.flat, ['c', 'l', 'bm', 'shadedrelief', 'etopo', 'etopo']):
    m = Basemap(projection='ortho', lat_0=45, lon_0=-100, resolution='l', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='coral', lake_color='aqua')
    m.drawmeridians(range(0, 360, 30))
    m.drawparallels(range(-90, 90, 30))

    for city, (lat, lon) in cities.items():
        x, y = m(lon, lat)
        m.plot(x, y, 'bo', markersize=5)
        ax.text(x, y, city, fontsize=8, ha='left')

    lons, lats = np.meshgrid(np.linspace(-180, 180, 100), np.linspace(-90, 90, 100))
    data = np.sin(lons) * np.cos(lats)
    x, y = m(lons, lats)
    m.contourf(x, y, data, cmap='coolwarm')

    m.drawmapboundary(fill_color='aqua')
    m.drawlsmask(land_color='green', ocean_color='aqua')
    m.bluemarble()
    m.shadedrelief()
    m.etopo()
    m.etopo(transparent=True)

plt.tight_layout()
plt.show()
```