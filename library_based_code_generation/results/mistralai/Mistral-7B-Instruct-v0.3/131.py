 Here is a simplified version of the code using Basemap and Matplotlib libraries. Note that you need to install these libraries using `pip install basemap matplotlib` before running the code. Also, the data for the cities and the elevation data are not included in this code.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.patches import Circle

cities = [(45, -100), (45, -120), (50, -100), (50, -120), (40, -110)]
city_names = ['City1', 'City2', 'City3', 'City4', 'City5']

elev_data = np.random.rand(100, 100)

fig = plt.figure(figsize=(10, 10))

for background in ['continent', 'ocean', 'blue_marble', 'shaded', 'etopo', 'etopo_transparent']:
    m = Basemap(projection='ortho', lat_0=45, lon_0=-100, resolution='i',
                ellps='WGS84')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color=background)
    m.drawmeridians(np.arange(-180, 181, 30))
    m.drawparallels(np.arange(-90, 91, 30))

    x, y = m(cities[:, 0], cities[:, 1])
    for i, (x_city, y_city) in enumerate(zip(x, y)):
        circle = Circle(xy=(x_city, y_city), radius=0.05, fill=True, ec='black')
        plt.gca().add_patch(circle)
        plt.text(x_city, y_city, city_names[i], ha='center')

    if background in ['shaded', 'etopo', 'etopo_transparent']:
        z = m(np.arange(-90, 91, 30))[2]
        X, Y = np.meshgrid(m.x, m.y)
        Z = np.interp(X, m.x, elev_data)
        Z[Y < m.ylim[0][0]] = -9999
        Z[Y > m.ylim[1][1]] = -9999
        CS = m.contourf(X, Y, Z, 10)
        plt.clabel(CS, inline=True, fontsize=10)

    plt.title(f'Background: {background}')
    plt.show()
```

This code creates six maps with different backgrounds and plots five cities on each map. It also contours some elevation data on the maps that have shaded relief, etopo, or etopo_transparent backgrounds. The cities are represented by filled circles, and their names are displayed next to the circles. The maps are displayed one by one.