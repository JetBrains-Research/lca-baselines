 ```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pysteps as ps
from pysteps.nowcast import STEPS
from pysteps.vis import plot_precip_field
from pysteps.utils import _create_motion_field, _convert_proj4_to_grid_mapping, _convert_grid_mapping_to_proj4

# Download and prepare data
ps.api._create_and_install_plugin('pysteps_data')
data_path = ps.api.download_mrms_data(date='20210601', n_files=1)
data = ps.api.data.read_gdal(data_path)
data = ps.api.to_reflectivity(data, method='zr')
data = ps.api.to_raindepth(data, method='zr')
data = ps.api.to_rainrate(data)

# Upscale data to 2 km resolution
data_upscaled = ps.api.upscale(data, method='nearest', factor=2)

# Handle missing values and log-transform data
data_upscaled[np.isnan(data_upscaled)] = 0
data_log = np.log(data_upscaled + 1)

# Estimate motion field
motion_field = _create_motion_field(data_log, method='lkm', radius_of_influence=10)

# Perform ensemble nowcast
nowcast_model = STEPS(motion_field=motion_field, n_ens=10, radius_of_influence=10)
nowcast_log = nowcast_model.nowcast(data_log, timesteps=12)

# Back-transform nowcast to rain rates
nowcast_rainrate = np.exp(nowcast_log) - 1

# Plot some of the realizations
plot_precip_field(nowcast_rainrate[:, :, 0], title='Nowcast realization 1')
plot_precip_field(nowcast_rainrate[:, :, 5], title='Nowcast realization 6')
plot_precip_field(nowcast_rainrate[:, :, 9], title='Nowcast realization 10')

# Verify probabilistic forecasts
verif_model = ps.api.Verification(data_log, nowcast_log)
verif_model.roc_curve()
verif_model.reliability_diagram()
verif_model.rank_histogram()
```
Please note that this is a simplified version of the code and may not cover all edge cases or specific requirements. It is recommended to adjust the code according to the specific needs and to test it thoroughly before using it in a production environment.