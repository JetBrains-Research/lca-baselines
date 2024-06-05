```python
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (MeanShift, MiniBatchKMeans, AgglomerativeClustering,
                             SpectralClustering, DBSCAN, Birch, AffinityPropagation)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
import warnings
import hdbscan
from sklearn.metrics import silhouette_score

# Generate datasets
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)
aniso = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
aniso = (np.dot(aniso[0], transformation), aniso[1])

datasets_list = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'damping': .75, 'preference': -220, 'n_clusters': 3}),
    (aniso, {'damping': .75, 'preference': -220, 'n_clusters': 3}),
    (blobs, {'damping': .75, 'preference': -220, 'n_clusters': 3}),
    (no_structure, {'damping': .75, 'preference': -220, 'n_clusters': 0})
]

clustering_algorithms = [
    ('MiniBatchKMeans', MiniBatchKMeans(n_clusters=3)),
    ('AffinityPropagation', AffinityPropagation(damping=.9, preference=-200)),
    ('MeanShift', MeanShift(bin_seeding=True)),
    ('SpectralClustering', SpectralClustering(n_clusters=3, affinity="nearest_neighbors")),
    ('AgglomerativeClustering', AgglomerativeClustering(n_clusters=3)),
    ('DBSCAN', DBSCAN(eps=.2)),
    ('HDBSCAN', hdbscan.HDBSCAN(min_cluster_size=15)),
    ('Birch', Birch(n_clusters=3)),
    ('GaussianMixture', GaussianMixture(n_components=3)),
]

fig, axes = plt.subplots(len(datasets_list), len(clustering_algorithms), figsize=(len(clustering_algorithms) * 2 + 3, len(datasets_list) * 2.5))
plt.subplots_adjust(bottom=.1, top=.9, wspace=.05, hspace=.3)

for i_dataset, (dataset, algo_params) in enumerate(datasets_list):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # Create clustering estimators
    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                        " may not work as expected.",
                category=UserWarning)
            algorithm.set_params(**algo_params)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        axes[i_dataset, clustering_algorithms.index((name, algorithm))].scatter(X[:, 0], X[:, 1], s=10, c=y_pred)
        axes[i_dataset, clustering_algorithms.index((name, algorithm))].set_title(f'{name}\n{t1 - t0:.2f}s', size=9)
        axes[i_dataset, clustering_algorithms.index((name, algorithm))].set_xticks(())
        axes[i_dataset, clustering_algorithms.index((name, algorithm))].set_yticks(())

plt.show()
```