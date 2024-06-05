 ```python
import numpy as np
import xarray as xr
import meteosteps as mts
import matplotlib.pyplot as plt
from scipy.stats import norm

# Read radar data
data = xr.open_dataset('radar_data.nc')
precipitation = data.precipitation

# Upscale to 2 km resolution
upscaled_precipitation = precipitation.interp(distance=np.arange(0, precipitation.distance.max(), 2000))

# Convert to rain rate
rain_rate = upscaled_precipitation * 60 * 60 / 1e3  # convert mm/s

# Log-transform data and handle missing values
rain_rate = np.log1p(rain_rate)
rain_rate[np.isnan(rain_rate)] = 0

# Estimate motion field
motion_field = mts.estimate_motion_field(rain_rate, method='lkm')

# Perform ensemble nowcast
nowcast = mts.ensemble_nowcast(rain_rate, motion_field, ens_size=100, method='steps')

# Back-transform nowcast to rain rates
nowcast = np.expm1(nowcast)

# Plot some realizations
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(nowcast[:, :, i], cmap='viridis', origin='upper')
    plt.colorbar()
    plt.title('Realization {}'.format(i+1))
plt.show()

# Verify probabilistic forecasts
verif_roc = mts.verify_roc(rain_rate, nowcast, obs_thresholds=np.arange(0.1, 10, 0.1))
verif_reliability = mts.verify_reliability(rain_rate, nowcast, nbins=10)
verif_rank_histogram = mts.verify_rank_histogram(rain_rate, nowcast, nbins=10)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(np.arange(0.1, 10, 0.1), verif_roc.roc_skill_score, label='ROC Skill Score')
plt.plot([0, 10], [0, 1], 'k--', label='Perfect forecast')
plt.xlabel('False alarm rate')
plt.ylabel('Hit rate')
plt.title('ROC curve')
plt.legend()
plt.show()

# Plot reliability diagrams
plt.figure(figsize=(6, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.plot(verif_reliability.bins[:-1], verif_reliability.reliability[i, :], label='Realization {}'.format(i+1))
    plt.plot([verif_reliability.bins[0], verif_reliability.bins[-1]], [i/10, i/10], 'k--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.title('Reliability diagram')
plt.legend()
plt.show()

# Plot rank histograms
plt.figure(figsize=(6, 6))
plt.hist(verif_rank_histogram.ranks.flatten(), bins=np.arange(-0.5, verif_rank_histogram.ranks.max()+1), density=True, alpha=0.5, label='Ensemble members')
plt.axvline(0.5, color='k', linestyle='--', label='Perfect forecast')
plt.xlabel('Rank')
plt.ylabel('Density')
plt.title('Rank histogram')
plt.legend()
plt.show()
```