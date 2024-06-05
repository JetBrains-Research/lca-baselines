```python
import numpy as np
import matplotlib.pyplot as plt
from pysteps import motion, nowcasts, io, visualization, verification
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.ndimage import gaussian_filter

# Download and read the radar data
date = "20230610"
time = "100000"
data_source = "mch"
root_path = io.archive.get_root_path(data_source)
filename = io.archive.find_by_date(date, time, root_path, "netcdf")[0]
R, _, metadata = io.import_mch_gif(filename)

# Upscale data to 2 km resolution
R = np.array(R)
R_upscaled = gaussian_filter(R, sigma=1)

# Convert to rain rate
R_upscaled = conversion.to_rainrate(R_upscaled, metadata)

# Log-transform and handle missing values
R_log, metadata = transformation.dB_transform(R_upscaled, metadata, threshold=0.1, zerovalue=-15)
R_log = np.where(np.isnan(R_log), -15, R_log)

# Estimate motion field
oflow_method = motion.get_method("LK")
UV = oflow_method(R_log)

# Perform ensemble nowcast using STEPS
n_ens_members = 20
n_leadtimes = 12  # Assuming 5-minute intervals, for 1 hour forecast
seed = 42  # For reproducibility
extrapolator = nowcasts.get_method("extrapolation")
R_fct = extrapolator(R_log, UV, n_leadtimes, n_ens_members, seed=seed)

# Back-transform to rain rates
R_fct_rainrate = np.empty_like(R_fct)
for i in range(n_ens_members):
    R_fct_rainrate[i], _ = transformation.dB_transform(R_fct[i], inverse=True)

# Plot some of the realizations
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axs.flatten()):
    plot_precip_field(R_fct_rainrate[i, -1], ax=ax, title=f"Realization {i+1}")
plt.tight_layout()
plt.show()

# Verification
obs = R_upscaled[-1]  # Assuming the last timestep of the upscaled data is the observation
fct = R_fct_rainrate[:, -1]  # Last timestep of each ensemble member

# ROC curve
obs_binary = np.where(obs > 1, 1, 0)  # Thresholding at 1 mm/h for binary classification
fct_probs = np.mean(fct > 1, axis=0)  # Probability of exceeding 1 mm/h
fpr, tpr, thresholds = roc_curve(obs_binary.flatten(), fct_probs.flatten())
roc_auc = roc_auc_score(obs_binary.flatten(), fct_probs.flatten())

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Reliability diagrams and rank histograms will require additional libraries or custom implementation
# This example focuses on the ROC curve for simplicity
```