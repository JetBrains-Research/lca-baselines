 Here is a simplified version of the Python code using sklearn and seaborn libraries. Please note that this is a basic structure and might need adjustments based on your specific requirements.

```python
import numpy as np
from sklearn.datasets import MakeBlobs, MakeMoons, make_blobs, make_circles
from sklearn.cluster import MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, HDBSCAN, OPTICS, AffinityPropagation, Birch, GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from time import time
import seaborn as sns
import matplotlib.pyplot as plt

datasets = [make_circles(n_samples=1000, factor=.5, noise=.1),
            make_moons(n_samples=1000, noise=.1),
            make_blobs(n_samples=1000, n_centers=5, centers=np.array([[0, 0], [2, 2], [-2, -2], [4, -2], [-4, 2]]), random_state=0),
            make_blobs(n_samples=1000, n_centers=5, centers=np.array([[0, 0], [2, 2], [-2, -2], [4, -2], [-4, 2]]), cluster_std=np.array([1, 2, 1, 3, 1]), random_state=0),
            make_blobs(n_samples=1000, n_centers=1, centers=[[0, 0]], cluster_std=np.array([1, 1, 1, 1, 1]), random_state=0)]

algorithms = [MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, HDBSCAN, OPTICS, AffinityPropagation, Birch, GaussianMixture]

fig, axs = plt.subplots(len(datasets), len(algorithms), figsize=(15, 20))

for i, dataset in enumerate(datasets):
    X = StandardScaler().fit_transform(dataset)
    start_time = time()
    for j, algorithm in enumerate(algorithms):
        try:
            kmeans = algorithm(n_clusters=len(dataset.centers_)).fit(X)
            elapsed_time = time() - start_time
            axs[i, j].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
            axs[i, j].set_title(algorithm.__name__ + ' - Time taken: ' + str(round(elapsed_time, 2)) + ' seconds')
        except Exception as e:
            print(f'Warning: {e} for {algorithm.__name__} on dataset {i}')

plt.show()
```

This code generates several datasets, applies various clustering algorithms, handles warnings, measures the time taken for each algorithm, and visualizes the results. However, it does not include the anisotropically distributed data and the dataset with no structure, as creating those datasets requires more complex methods not provided by sklearn's built-in datasets. You can find examples of generating those datasets online. Also, this code does not handle the kneighbors_graph warning for DBSCAN and HDBSCAN, as it is a common issue when using those algorithms and can be ignored in many cases.