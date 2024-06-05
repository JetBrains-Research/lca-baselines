```python
import pysteps
from pysteps import motion, nowcasts, verification
from pysteps.utils import conversion
import numpy as np

# Read precipitation field data
data = pysteps.io.import_mch_gif("path_to_data")

# Upscale to 2 km resolution
data = pysteps.utils.griddata.reproject_data(data, geodata=None, proj="epsg:4326", hres=2000, vres=2000)

# Convert to rain rate
data = pysteps.utils.conversion.to_rainrate(data, 5, threshold=0.1)

# Log-transform the data
data = np.log(data)

# Handle missing values
data[np.isnan(data)] = 0

# Estimate the motion field
motion_field = pysteps.motion.get_motion_field(data)

# Perform ensemble nowcast using the STEPS approach
nowcast = nowcasts.ensemble_nowcast(data, motion_field, n_ens_members=10)

# Back-transform the nowcast to rain rates
nowcast = np.exp(nowcast)

# Plot some of the realizations
pysteps.visualization.plot_precip_field(nowcast[0])

# Verify the probabilistic forecasts
verification.plot_roc_curve(data, nowcast)
verification.plot_reliability_diagram(data, nowcast)
verification.plot_rank_histogram(data, nowcast)
```