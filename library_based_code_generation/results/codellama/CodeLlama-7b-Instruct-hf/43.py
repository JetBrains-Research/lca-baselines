  import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import STEPS
from pyproj import Proj

# Read precipitation field data
data = np.loadtxt('precipitation_field_data.txt')

# Upscale data to 2 km resolution
data_2km = np.zeros((data.shape[0], data.shape[1]*2, data.shape[2]*2))
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            data_2km[i, j*2, k*2] = data[i, j, k]

# Convert data to rain rate
data_2km = data_2km * 1000 / 60 / 60

# Log-transform data and handle missing values
data_2km[data_2km == 0] = np.nan
data_2km = np.log(data_2km)

# Estimate motion field
motion_field = np.zeros((data_2km.shape[0], data_2km.shape[1], data_2km.shape[2], 2))
for i in range(data_2km.shape[0]):
    for j in range(data_2km.shape[1]):
        for k in range(data_2km.shape[2]):
            motion_field[i, j, k, 0] = np.random.normal(0, 1)
            motion_field[i, j, k, 1] = np.random.normal(0, 1)

# Perform ensemble nowcast using STEPS
steps = STEPS(n_estimators=100, random_state=0)
steps.fit(data_2km, motion_field)
nowcast = steps.predict(data_2km)

# Back-transform nowcast to rain rates
nowcast = np.exp(nowcast)
nowcast = nowcast * 60 * 60 / 1000

# Plot some of the realizations
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(nowcast[i, :, :], cmap='gray')
    plt.axis('off')

# Verify probabilistic forecasts using ROC curve, reliability diagrams, and rank histograms
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(data_2km, nowcast)
roc_auc = auc(fpr, tpr)

# Compute reliability diagrams
reliability_diagrams = np.zeros((data_2km.shape[0], data_2km.shape[1], data_2km.shape[2], 2))
for i in range(data_2km.shape[0]):
    for j in range(data_2km.shape[1]):
        for k in range(data_2km.shape[2]):
            reliability_diagrams[i, j, k, 0] = np.nanmean(nowcast[:, i, j, k])
            reliability_diagrams[i, j, k, 1] = np.nanstd(nowcast[:, i, j, k])

# Compute rank histograms
rank_histograms = np.zeros((data_2km.shape[0], data_2km.shape[1], data_2km.shape[2], 2))
for i in range(data_2km.shape[0]):
    for j in range(data_2km.shape[1]):
        for k in range(data_2km.shape[2]):
            rank_histograms[i, j, k, 0] = np.nanmean(np.argsort(nowcast[:, i, j, k]))
            rank_histograms[i, j, k, 1] = np.nanstd(np.argsort(nowcast[:, i, j, k]))

# Plot ROC curve
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')

# Plot reliability diagrams
plt.figure(figsize=(10, 10))
for i in range(data_2km.shape[0]):
    for j in range(data_2km.shape[1]):
        for k in range(data_2km.shape[2]):
            plt.subplot(2, 5, i*data_2km.shape[1]*data_2km.shape[2] + j*data_2km.shape[2] + k + 1)
            plt.imshow(reliability_diagrams[i, j, k, :], cmap='gray')
            plt.axis('off')

# Plot rank histograms
plt.figure(figsize=(10, 10))
for i in range(data_2km.shape[0]):
    for j in range(data_2km.shape[1]):
        for k in range(data_2km.shape[2]):
            plt.subplot(2, 5, i*data_2km.shape[1]*data_2km.shape[2] + j*data_2km.shape[2] + k + 1)
            plt.imshow(rank_histograms[i, j, k, :], cmap='gray')
            plt.axis('off')