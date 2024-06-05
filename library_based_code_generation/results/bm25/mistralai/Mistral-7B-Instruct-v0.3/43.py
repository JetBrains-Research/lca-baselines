 Here is a simplified Python code example using the Pysteps library for the task. Please note that this is a basic example and might need adjustments to fit your specific needs.

```python
import pysteps
import numpy as np
import matplotlib.pyplot as plt

# Install plugins
pysteps._create_and_install_plugin('pysteps_radar')
pysteps._create_and_install_plugin('pysteps_nowcast')

# Download MeteoSwiss radar data
pysteps.download_pysteps_data('radar_data.h5')

# Load radar data
radar_data = pysteps.data.RadarData('radar_data.h5')

# Upscale data to 2 km resolution
upscaled_data = pysteps._convert_grid_mapping_to_proj4(radar_data.grid_mapping, 2000)
upscaled_data = pysteps.data.RadarData(upscaled_data, radar_data.time)

# Convert data to reflectivity, rain depth, and rain rate
reflectivity = upscaled_data.to_reflectivity()
rain_depth = upscaled_data.to_raindepth()
rain_rate = upscaled_data.to_rainrate()

# Log-transform data and handle missing values
rain_rate_log = np.log1p(rain_rate)
rain_rate_log[rain_rate_log == -np.inf] = 0

# Create motion field
motion_field = pysteps._create_motion_field(rain_rate_log, method='STEPS')

# Perform ensemble nowcast
nowcast_data = pysteps.nowcast_main_loop(rain_rate_log, motion_field, ensemble_size=10)

# Back-transform nowcast to rain rates
nowcast_rain_rate = np.exp(nowcast_data) - 1

# Plot some realizations
for i in range(10):
    pysteps._plot_field(nowcast_rain_rate[i], title=f'Nowcast realization {i+1}')

# Verify probabilistic forecasts
# Calculate ROC curve, reliability diagrams, and rank histograms
# ... (This part is complex and requires specific implementation)
```

This code downloads MeteoSwiss radar data, upscales it, converts it to rain rate, performs an ensemble nowcast using the STEPS approach, and plots some of the realizations. However, the verification part (ROC curve, reliability diagrams, and rank histograms) is not implemented in this example. You would need to implement those parts according to your specific requirements.