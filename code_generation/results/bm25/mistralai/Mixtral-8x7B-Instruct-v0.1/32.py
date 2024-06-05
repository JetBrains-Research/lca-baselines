 import time
import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import make\_blobs, make\_circles, make\_moons
from sklearn.cluster import MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, HDBSCAN, OPTICS, AffinityPropagation, Birch, GaussianMixture
from sklearn.neighbors import kneighbors\_graph
from sklearn.metrics import calinski\_harabasz\_score

datasets = [make\_circles(n\_samples=300, factor=.5, noise=.05),
make\_moons(n\_samples=300, noise=.05),
make\_blobs(n\_samples=300, centers=4, random\_state=0, cluster\_std=1.0),
make\_blobs(n\_samples=300, centers=4, random\_state=0, cluster\_std=[1.0, 0.5, 0.5, 0.5]),
generate\_dataset(n\_samples=300, noise=.05),
None]

clustering\_algorithms = [
[MeanShift, {'bandwidth': 0.8}],
[MiniBatchKMeans, {'n\_clusters': 4, 'batch\_size': 100}],
[AgglomerativeClustering, {'n\_clusters': 4}],
[SpectralClustering, {'n\_clusters': 4, 'assign\_labels': 'discretize'}],
[DBSCAN, {'eps': 0.3, 'min\_samples': 10}],
[HDBSCAN, {'min\_cluster\_size': 5, 'min\_samples': 5, 'alpha': 1.0, 'cluster\_selection\_epsilon': 0.0}],
[OPTICS, {'min\_samples': 5, 'xi': 0.05, 'min\_cluster\_size': 5}],
[AffinityPropagation, {'damping': 0.5, 'convergence\_iter': 20}],
[Birch, {'threshold': 0.05, 'branching\_factor': 50}],
[GaussianMixture, {'n\_components': 4, 'covariance\_type': 'full'}]
]

times = []
for data in datasets:
warnings.filterwarnings("ignore", category=UserWarning)
t0 = time.time()
X = data if data is not None else np.random.rand(300, 2)
kng = kneighbors\_graph(X, n\_neighbors=10, include\_self=False)
clustering\_results = []
for algorithm in clustering\_algorithms:
clustering\_algorithms[clustering\_algorithms.index(algorithm)][0]().fit(X)
clustering\_results.append(clustering\_algorithms[clustering\_algorithms.index(algorithm)][0]().labels_)
times.append(time.time() - t0)
if data is not None:
calinski\_harabasz\_score(X, clustering\_results[-1])
print("Calinski-Harabasz Score:", calinski\_harabasz\_score(X, clustering\_results[-1]))

fig, axs = plt.subplots(nrows=2, ncols=len(clustering\_algorithms), figsize=(20, 8))
for i, result in enumerate(clustering\_results):
axs[0, i].scatter(X[:, 0], X[:, 1], c=result)
axs[0, i].set\_title(clustering\_algorithms[i][0].\_\_name\_\_)
axs[1, i].plot([1000 * (i+1) for i in range(len(times))], times, marker='o')
axs[1, i].set\_xlabel('Algorithm')
axs[1, i].set\_ylabel('Time (ms)')
plt.show()