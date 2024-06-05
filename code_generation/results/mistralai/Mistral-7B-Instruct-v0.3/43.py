 Here is a simplified Python code using the `xarray` and `metpy` libraries for the task. Note that this is a basic example and might need adjustments to fit your specific needs.

```python
import xarray as xr
import metpy.calc as mpcalc
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.met1d import interpolate_1d

# Load data
data = xr.open_dataset('radar_data.nc')

# Upscale data to 2 km resolution
upsampled_data = data.interp(time=data.time.resample(time='2H').mean('time'))

# Convert precipitation to rain rate (mm/h)
upsampled_data['precipitation'] = upsampled_data['precipitation'] * 2 * 3600

# Log-transform data and handle missing values
upsampled_data['precipitation'] = np.log1p(upsampled_data['precipitation'].fillna(1))

# Estimate motion field
# This part requires a wind estimation method, e.g., using MeteoSwiss's wind analysis data
motion_field = estimate_motion_field()

# Perform ensemble nowcast using STEPS approach
ensemble_size = 10
nowcast_data = upsampled_data.copy()
for _ in range(ensemble_size):
    nowcast_data['precipitation'] *= np.exp(np.random.normal(0, 1, size=nowcast_data.shape))
    nowcast_data = nowcast_data.assign_coords(time=nowcast_data.time + motion_field['time_step'])

# Back-transform nowcast to rain rates
nowcast_data['precipitation'] = np.expm1(nowcast_data['precipitation']) * 2 * 3600

# Plot some realizations
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i in range(ensemble_size):
    ax[i].imshow(nowcast_data.sel(time=nowcast_data.time[i]).precipitation.values, cmap='viridis')
    ax[i].set_title(f'Nowcast realization {i+1}')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.show()

# Verify probabilistic forecasts
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from metpy.plots import reliability_diagram, rank_histogram

# Calculate verification metrics
y_true = data['precipitation'].values > 0
y_pred = nowcast_data.sel(time=nowcast_data.time[-1]).precipitation.values > 0
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label='ROC')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot reliability diagram
reliability_diagram(y_true, y_pred, bins=10)
plt.show()

# Plot rank histogram
rank_histogram(y_true, y_pred, bins=10)
plt.show()
```

This code assumes that you have a NetCDF file containing the precipitation data from MeteoSwiss radar. It also assumes that you have a method for estimating the motion field and that you have the necessary libraries installed. Adjust the code as needed to fit your specific data and requirements.