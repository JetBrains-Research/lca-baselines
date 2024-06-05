```python
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import (MeanShift, MiniBatchKMeans, AgglomerativeClustering,
                             SpectralClustering, DBSCAN, Birch, AffinityPropagation)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
try:
    import hdbscan
except ImportError:
    hdbscan = None
try:
    from sklearn.cluster import OPTICS
except ImportError:
    OPTICS = None

# Generate datasets
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240, 'quantile': .2, 'n_clusters': 2}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'quantile': .2, 'n_clusters': 2}),
    (varied, {'damping': .75, 'preference': -220, 'quantile': .2, 'n_clusters': 3}),
    (aniso, {'damping': .75, 'preference': -220, 'quantile': .2, 'n_clusters': 3}),
    (blobs, {'damping': .75, 'preference': -220, 'quantile': .2, 'n_clusters': 3}),
    (no_structure, {'damping': .75, 'preference': -220, 'quantile': .2, 'n_clusters': 0})
]

clustering_algorithms = [
    ('MiniBatchKMeans', MiniBatchKMeans(n_clusters=3)),
    ('AffinityPropagation', AffinityPropagation(damping=.9, preference=-200)),
    ('MeanShift', MeanShift(bin_seeding=True)),
    ('SpectralClustering', SpectralClustering(n_clusters=3, affinity="nearest_neighbors")),
    ('AgglomerativeClustering', AgglomerativeClustering(n_clusters=3)),
    ('DBSCAN', DBSCAN(eps=0.3, min_samples=10)),
    ('OPTICS', OPTICS(min_samples=10, xi=.05, min_cluster_size=.1) if OPTICS else None),
    ('Birch', Birch(n_clusters=3)),
    ('GaussianMixture', GaussianMixture(n_components=3, covariance_type='full')),
    ('HDBSCAN', hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10) if hdbscan else None)
]

fig, axes = plt.subplots(len(datasets), len(clustering_algorithms) + 1, figsize=(len(clustering_algorithms) * 2 + 3, len(datasets) * 2))
plt.subplots_adjust(bottom=.001, top=.96, left=.001, right=.99, wspace=.05, hspace=.01)

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # Plot original data
    axes[i_dataset, 0].scatter(X[:, 0], X[:, 1], s=10)
    axes[i_dataset, 0].set_title("Original Data")
    for name, algorithm in clustering_algorithms:
        if algorithm is None:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="the number of connected components of the connectivity matrix is [0-9]+ > 1. Completing it to avoid stopping the tree early.", category=UserWarning)
            warnings.filterwarnings("ignore", message="Graph is not fully connected, spectral clustering will not work as expected.", category=UserWarning)
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

        axes[i_dataset, clustering_algorithms.index((name, algorithm)) + 1].scatter(X[:, 0], X[:, 1], c=y_pred, s=10)
        axes[i_dataset, clustering_algorithms.index((name, algorithm)) + 1].set_title(name + "\n%.2fs" % (t1 - t0))

plt.show()
```