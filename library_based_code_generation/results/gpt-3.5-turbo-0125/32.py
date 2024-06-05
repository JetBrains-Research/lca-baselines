```python
# Import necessary libraries
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, AffinityPropagation, Birch, GaussianMixture
from sklearn.cluster import OPTICS, hdbscan
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

# Generate toy datasets
n_samples = 1500
noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = make_moons(n_samples=n_samples, noise=.05)
blobs = make_blobs(n_samples=n_samples, random_state=8)
aniso = (datasets.make_blobs(n_samples=n_samples, random_state=170))[0]
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5])[0]
no_structure = np.random.rand(n_samples, 2), None

datasets = [noisy_circles, noisy_moons, blobs, aniso, varied, no_structure]

# Set up parameters for clustering algorithms
clustering_algorithms = [
    ('MeanShift', MeanShift()),
    ('MiniBatchKMeans', MiniBatchKMeans(n_clusters=3)),
    ('AgglomerativeClustering', AgglomerativeClustering(n_clusters=3)),
    ('SpectralClustering', SpectralClustering(n_clusters=3)),
    ('DBSCAN', DBSCAN(eps=0.3)),
    ('HDBSCAN', hdbscan.HDBSCAN(min_cluster_size=15)),
    ('OPTICS', OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)),
    ('AffinityPropagation', AffinityPropagation()),
    ('Birch', Birch(n_clusters=3)),
    ('GaussianMixture', GaussianMixture(n_components=3))
]

# Handle warnings related to kneighbors_graph
warnings.filterwarnings('ignore')

# Apply clustering algorithms to each dataset and measure time taken
for dataset in datasets:
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)

    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        plt.scatter(X[:, 0], X[:, 1], c=algorithm.labels_, cmap='viridis')
        plt.title(f'{name} on {dataset.__class__.__name__}')
        plt.text(0.99, 0.01, f'{t1 - t0:.2f}s', transform=plt.gca().transAxes, horizontalalignment='right')
        plt.show()
```