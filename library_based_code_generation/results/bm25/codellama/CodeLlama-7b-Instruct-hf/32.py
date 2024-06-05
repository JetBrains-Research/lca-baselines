  import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, HDBSCAN, OPTICS, AffinityPropagation, Birch, GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles, make_anisotropic, make_blobs_varied, make_no_structure
from sklearn.metrics import silhouette_score

# Generate datasets
datasets = {
    'noisy circles': make_circles(noise=0.2),
    'noisy moons': make_moons(noise=0.3),
    'blobs': make_blobs(n_samples=100, n_features=2, centers=5, cluster_std=0.5),
    'anisotropic': make_anisotropic(n_samples=100, n_features=2, n_classes=5, random_state=42),
    'blobs with varied variances': make_blobs_varied(n_samples=100, n_features=2, centers=5, cluster_std=[0.5, 1, 2, 0.5, 1.5]),
    'no structure': make_no_structure(n_samples=100, n_features=2, n_classes=5, random_state=42)
}

# Set up parameters for clustering
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
max_iter = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Apply clustering algorithms to each dataset
for dataset_name, dataset in datasets.items():
    print(f'Dataset: {dataset_name}')
    for n_cluster in n_clusters:
        print(f'Number of clusters: {n_cluster}')
        for max_iter in max_iter:
            print(f'Maximum number of iterations: {max_iter}')
            # MeanShift
            ms = MeanShift(n_clusters=n_cluster, max_iter=max_iter)
            ms.fit(dataset)
            print(f'MeanShift: {ms.labels_}')
            # MiniBatchKMeans
            mbk = MiniBatchKMeans(n_clusters=n_cluster, max_iter=max_iter)
            mbk.fit(dataset)
            print(f'MiniBatchKMeans: {mbk.labels_}')
            # AgglomerativeClustering
            ac = AgglomerativeClustering(n_clusters=n_cluster, linkage='ward', max_iter=max_iter)
            ac.fit(dataset)
            print(f'AgglomerativeClustering: {ac.labels_}')
            # SpectralClustering
            sc = SpectralClustering(n_clusters=n_cluster, affinity='nearest_neighbors', n_neighbors=10, max_iter=max_iter)
            sc.fit(dataset)
            print(f'SpectralClustering: {sc.labels_}')
            # DBSCAN
            db = DBSCAN(eps=0.5, min_samples=10, metric='euclidean', max_iter=max_iter)
            db.fit(dataset)
            print(f'DBSCAN: {db.labels_}')
            # HDBSCAN
            hdb = HDBSCAN(min_cluster_size=10, min_samples=10, metric='euclidean', max_iter=max_iter)
            hdb.fit(dataset)
            print(f'HDBSCAN: {hdb.labels_}')
            # OPTICS
            opt = OPTICS(min_samples=10, metric='euclidean', max_iter=max_iter)
            opt.fit(dataset)
            print(f'OPTICS: {opt.labels_}')
            # AffinityPropagation
            ap = AffinityPropagation(damping=0.5, max_iter=max_iter)
            ap.fit(dataset)
            print(f'AffinityPropagation: {ap.labels_}')
            # Birch
            br = Birch(n_clusters=n_cluster, threshold=0.5, max_iter=max_iter)
            br.fit(dataset)
            print(f'Birch: {br.labels_}')
            # GaussianMixture
            gm = GaussianMixture(n_components=n_cluster, max_iter=max_iter)
            gm.fit(dataset)
            print(f'GaussianMixture: {gm.labels_}')

# Visualize results
for dataset_name, dataset in datasets.items():
    print(f'Dataset: {dataset_name}')
    for n_cluster in n_clusters:
        print(f'Number of clusters: {n_cluster}')
        for max_iter in max_iter:
            print(f'Maximum number of iterations: {max_iter}')
            # MeanShift
            ms = MeanShift(n_clusters=n_cluster, max_iter=max_iter)
            ms.fit(dataset)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=ms.labels_)
            plt.title(f'MeanShift: {n_cluster} clusters')
            plt.show()
            # MiniBatchKMeans
            mbk = MiniBatchKMeans(n_clusters=n_cluster, max_iter=max_iter)
            mbk.fit(dataset)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=mbk.labels_)
            plt.title(f'MiniBatchKMeans: {n_cluster} clusters')
            plt.show()
            # AgglomerativeClustering
            ac = AgglomerativeClustering(n_clusters=n_cluster, linkage='ward', max_iter=max_iter)
            ac.fit(dataset)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=ac.labels_)
            plt.title(f'AgglomerativeClustering: {n_cluster} clusters')
            plt.show()
            # SpectralClustering
            sc = SpectralClustering(n_clusters=n_cluster, affinity='nearest_neighbors', n_neighbors=10, max_iter=max_iter)
            sc.fit(dataset)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=sc.labels_)
            plt.title(f'SpectralClustering: {n_cluster} clusters')
            plt.show()
            # DBSCAN
            db = DBSCAN(eps=0.5, min_samples=10, metric='euclidean', max_iter=max_iter)
            db.fit(dataset)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=db.labels_)
            plt.title(f'DBSCAN: {n_cluster} clusters')
            plt.show()
            # HDBSCAN
            hdb = HDBSCAN(min_cluster_size=10, min_samples=10, metric='euclidean', max_iter=max_iter)
            hdb.fit(dataset)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=hdb.labels_)
            plt.title(f'HDBSCAN: {n_cluster} clusters')
            plt.show()
            # OPTICS
            opt = OPTICS(min_samples=10, metric='euclidean', max_iter=max_iter)
            opt.fit(dataset)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=opt.labels_)
            plt.title(f'OPTICS: {n_cluster} clusters')
            plt.show()
            # AffinityPropagation
