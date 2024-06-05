 ```python
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification
from sklearn.cluster import MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, HDBSCAN, OPTICS, AffinityPropagation, Birch, GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

datasets = {
    "noisy_circles": make_circles(n_samples=1000, factor=.5, noise=.05),
    "noisy_moons": make_moons(n_samples=1000, noise=.05),
    "blobs": make_blobs(n_samples=1000, centers=4, random_state=0, cluster_std=1.0),
    "anisotropic_blobs": make_blobs(n_samples=1000, centers=4, random_state=0, cluster_std=[1.0, 2.5, 1.5, 3.0]),
    "varied_blobs": make_blobs(n_samples=1000, centers=4, random_state=0, cluster_std=[1.0, 2.0, 0.5, 1.5]),
    "no_structure": make_classification(n_samples=1000, n_features=20, n_informative=0, n_redundant=10, random_state=1)
}

datasets = {name: StandardScaler().fit_transform(data) for name, data in datasets.items()}

clustering_algorithms = {
    "MeanShift": MeanShift(),
    "MiniBatchKMeans": MiniBatchKMeans(n_clusters=4, random_state=0),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=4, linkage="ward"),
    "SpectralClustering": SpectralClustering(n_clusters=4, affinity="nearest_neighbors"),
    "DBSCAN": DBSCAN(eps=.3, min_samples=10),
    "HDBSCAN": HDBSCAN(min_cluster_size=10, gen_min_span_tree=True),
    "OPTICS": OPTICS(min_samples=10, xi=.05),
    "AffinityPropagation": AffinityPropagation(damping=0.5, max_iter=2000, random_state=None),
    "Birch": Birch(n_clusters=4, threshold=0.01),
    "GaussianMixture": GaussianMixture(n_components=4, random_state=0)
}

times = {}

for name, data in datasets.items():
    print(f"Processing dataset: {name}")
    times[name] = {}
    for algo_name, algo in clustering_algorithms.items():
        print(f"Processing algorithm: {algo_name}")
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            t0 = time.time()
            kneighbors_graph(data, n_neighbors=10, include_self=False)
            t1 = time.time()
            algo.fit(data)
            t2 = time.time()
        times[name][algo_name] = {"fit": t2 - t1, "kneighbors_graph": t1 - t0}

fig, axs = plt.subplots(nrows=len(clustering_algorithms), figsize=(10, 15))
sns.set_style("whitegrid")

for i, (algo_name, algo) in enumerate(clustering_algorithms.items()):
    ax = sns.heatmap(times, cmap="YlGnBu", yticklabels=list(datasets.keys()), xticklabels=["fit", "kneighbors_graph"], ax=axs[i])
    ax.set_title(algo_name)

plt.tight_layout()
plt.show()
```