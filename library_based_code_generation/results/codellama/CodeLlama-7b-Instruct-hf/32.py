  import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, HDBSCAN, OPTICS, AffinityPropagation, Birch, GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.utils.graph import kneighbors_graph

# Generate toy datasets
noisy_circles = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=0)[0]
noisy_moons = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=1)[0]
blobs = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=2)[0]
anisotropic = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=3, n_features=10)[0]
varied = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=4, n_features=10, variances=[1, 10, 100])[0]
no_structure = np.random.rand(100, 10)

# Set up parameters for clustering
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
max_iter = [100, 200, 300, 400, 500, 600, 700, 800, 900]

# Apply clustering algorithms to each dataset
results = {}
for dataset in [noisy_circles, noisy_moons, blobs, anisotropic, varied, no_structure]:
    results[dataset] = {}
    for algorithm in [MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, HDBSCAN, OPTICS, AffinityPropagation, Birch, GaussianMixture]:
        results[dataset][algorithm.__name__] = []
        for n_cluster in n_clusters:
            for max_iter in max_iter:
                try:
                    algorithm_instance = algorithm(n_clusters=n_cluster, max_iter=max_iter)
                    algorithm_instance.fit(dataset)
                    results[dataset][algorithm.__name__].append((n_cluster, max_iter, algorithm_instance.inertia_))
                except:
                    results[dataset][algorithm.__name__].append((n_cluster, max_iter, np.nan))

# Measure time taken for each algorithm to fit the data
for dataset in results:
    for algorithm in results[dataset]:
        results[dataset][algorithm] = np.array(results[dataset][algorithm])
        results[dataset][algorithm] = results[dataset][algorithm][results[dataset][algorithm][:, 2].argsort()]

# Visualize the results
for dataset in results:
    fig, axs = plt.subplots(nrows=len(results[dataset]), ncols=1, figsize=(10, 10))
    for i, algorithm in enumerate(results[dataset]):
        axs[i].plot(results[dataset][algorithm][:, 0], results[dataset][algorithm][:, 1], label=algorithm)
        axs[i].set_xlabel('Number of clusters')
        axs[i].set_ylabel('Inertia')
        axs[i].set_title(dataset)
        axs[i].legend()
    plt.show()