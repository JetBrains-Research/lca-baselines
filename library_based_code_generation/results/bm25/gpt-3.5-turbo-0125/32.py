```python
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification
from sklearn.cluster import MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, AffinityPropagation, Birch, GaussianMixture
from sklearn.cluster import OPTICS, hdbscan
import warnings
import time
import matplotlib.pyplot as plt

# Generate datasets
noisy_circles = make_circles(n_samples=1000, factor=0.5, noise=0.05)
noisy_moons = make_moons(n_samples=1000, noise=0.1)
blobs = make_blobs(n_samples=1000, random_state=8)
aniso = (make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1))
varied = make_blobs(n_samples=1000, cluster_std=[1.0, 2.5, 0.5])
no_structure = np.random.rand(1000, 2), None

datasets = [noisy_circles, noisy_moons, blobs, aniso, varied, no_structure]

# Clustering algorithms
clustering_algorithms = [
    ('MeanShift', MeanShift()),
    ('MiniBatchKMeans', MiniBatchKMeans()),
    ('AgglomerativeClustering', AgglomerativeClustering()),
    ('SpectralClustering', SpectralClustering()),
    ('DBSCAN', DBSCAN()),
    ('HDBSCAN', hdbscan.HDBSCAN()),
    ('OPTICS', OPTICS()),
    ('AffinityPropagation', AffinityPropagation()),
    ('Birch', Birch()),
    ('GaussianMixture', GaussianMixture())
]

# Handle warnings
warnings.filterwarnings('ignore')

# Fit data and measure time
for dataset in datasets:
    X, y = dataset
    for name, algorithm in clustering_algorithms:
        start_time = time.time()
        algorithm.fit(X)
        end_time = time.time()
        print(f'{name} on dataset: {end_time - start_time} seconds')

# Visualize results
fig, axs = plt.subplots(len(datasets), len(clustering_algorithms), figsize=(20, 20))
for i, dataset in enumerate(datasets):
    X, y = dataset
    for j, (name, algorithm) in enumerate(clustering_algorithms):
        algorithm.fit(X)
        axs[i, j].scatter(X[:, 0], X[:, 1], c=algorithm.labels_, cmap='viridis')
        axs[i, j].set_title(f'{name}')
plt.show()
```