import pandas as pd
import numpy as np


def np_rows(arr):
    return np.shape(arr)[0]


def np_cols(arr):
    return np.shape(arr)[1]


class KMeans:

    def __init__(self, n_clusters, distance,
                 tolerance=0.0001, max_iter=100, random_state=None):
        self.k = n_clusters
        self.calc_distance = distance

        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_state = random_state

        self.final_centroids = None

    def _initialize(self, data):
        arr = np.arange(np_rows(data))
        np.random.seed(self.random_state)
        choice = np.random.choice(arr, size=self.k, replace=False)

        return data[choice]

    def _assign_to_clusters(self, rows, centroids):

        distances = np.concatenate([self.calc_distance(rows, c, axis=1) for c in centroids]) \
            .reshape(np_rows(centroids), np_rows(rows))

        return np.argmin(distances, axis=0), np.sum(distances, axis=1).sum()

    def _find_new_centroids(self, data, assignments):
        centroids = []
        for c in np.unique(assignments):
            in_cluster = np.delete(data, np.where(assignments != c), axis=0)
            centroids.append(np.mean(in_cluster, axis=0))

        return np.array(centroids) \
            .reshape(self.k, np_cols(data))

    def fit(self, df: pd.DataFrame):
        np_df = df.to_numpy()

        centroids = self._initialize(np_df)
        inertia = None

        print(centroids)

        for i in range(self.max_iter):
            assignments, _inertia = self._assign_to_clusters(np_df, centroids)
            _centroids = self._find_new_centroids(np_df, assignments)

            if inertia is not None \
                    and (inertia - _inertia) < self.tolerance:
                print('Finished: Inertia change lower than tolerance')
                break

            if np.array_equal(centroids, _centroids):
                print('Finished: No changes in centroids')
                break

            inertia = _inertia
            centroids = _centroids

            print(f'Iteration {i + 1} finished')

        print("Finished.")
        self.final_centroids = centroids

    def predict(self, df: pd.DataFrame):
        if self.final_centroids is None:
            raise RuntimeError('fit must be called before predict')

        return self._assign_to_clusters(df.to_numpy(), self.final_centroids)[0]
