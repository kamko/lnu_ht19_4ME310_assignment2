import numpy as np

from src.log_util import log
from src.numpy_util import np_cols
from src.numpy_util import np_rows
from src.numpy_util import random_from_range
from src.numpy_util import to_numpy


class KMeans:

    def __init__(self, n_clusters, distance,
                 tolerance=0.0001, max_iter=100,
                 random_state=None,
                 verbose=True):
        self.k = n_clusters
        self.calc_distance = distance

        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self.final_centroids = None
        self.inertia = None

    def _initialize(self, data):
        choice = random_from_range(self.k, np_rows(data), self.random_state)
        return data[choice]

    def _assign_to_clusters(self, rows, centroids):

        distances = np.concatenate([self.calc_distance(rows, c, axis=1) for c in centroids]) \
            .reshape(np_rows(centroids), np_rows(rows))

        return np.argmin(distances, axis=0), np.min(distances, axis=0).sum()

    def _find_new_centroids(self, data, assignments):
        centroids = []
        for c in np.unique(assignments):
            in_cluster = np.delete(data, np.where(assignments != c), axis=0)
            centroids.append(np.mean(in_cluster, axis=0))

        return np.array(centroids) \
            .reshape(self.k, np_cols(data))

    def fit(self, data):
        data = to_numpy(data)

        centroids = self._initialize(data)
        inertia = None

        for i in range(self.max_iter):
            assignments, _inertia = self._assign_to_clusters(data, centroids)
            _centroids = self._find_new_centroids(data, assignments)

            if inertia is not None \
                    and (np.abs(inertia - _inertia)) < self.tolerance:
                log('Finished: Inertia change lower than tolerance', self.verbose)
                break

            if np.array_equal(centroids, _centroids):
                log('Finished: No changes in centroids', self.verbose)
                break

            inertia = _inertia
            centroids = _centroids

            log(f'Iteration {i + 1} finished. Inertia: {inertia}', self.verbose)

        log("Finished.", self.verbose)
        self.final_centroids = centroids
        self.inertia = inertia

    def predict(self, data):
        if self.final_centroids is None:
            raise RuntimeError('fit must be called before predict')

        return self._assign_to_clusters(to_numpy(data), self.final_centroids)[0]
