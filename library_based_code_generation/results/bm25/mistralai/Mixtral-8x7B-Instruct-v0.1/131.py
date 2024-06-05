 import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid, geos_data as GD, cm
import numpy as np

# Set up figure canvas
fig = plt.figure(figsize=(10, 8))

# Set up projection and region
projection = 'ortho'
llcrnrlat = 20
urcrnrlat = 70
llcrnrlon = -140
urcrnrlon = -60

for i in range(6):
    ax = fig.add_subplot(2, 3, i+1)

