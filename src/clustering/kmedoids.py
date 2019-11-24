import pandas as pd
import numpy as np

import itertools


class KMedoids:

    def __init__(self,
                 n_clusters,
                 distance_metric,
                 max_iter=100,
                 random_state=None):
        self.k = n_clusters
        self.calc_distance = distance_metric
        self.max_iter = max_iter
        self.random_state = random_state

        self.final_medoids = None

    def _initialize_random_medoids(self, df):
        index = df.index.to_series()
        medoids = index.sample(self.k, random_state=self.random_state)

        return medoids.to_list()

    def _assign_to_clusters(self, df, medoids_idx):
        np_df = df.to_numpy()

        not_medoids = np.delete(df.to_numpy(), medoids_idx, axis=0)
        medoids = np_df[medoids_idx]

        res = []
        for i, v in zip(medoids_idx, medoids):
            res.append(self.calc_distance(not_medoids, v, axis=1))

        cost_table = np.concatenate([i for i in res]) \
            .reshape(len(medoids_idx), len(not_medoids))
        return cost_table

    def _calculate_cost(self, train_df, medoids_idx):
        cost_table = self._assign_to_clusters(train_df, medoids_idx)
        return np.min(cost_table, axis=1).sum(-1)

    def fit(self, train_df: pd.DataFrame):
        best_medoids_idx = self._initialize_random_medoids(train_df)
        best_cost = self._calculate_cost(train_df, best_medoids_idx)

        iteration = 1
        while True:
            _non_medoids_idx = np.delete(train_df.index.to_numpy(), best_medoids_idx)
            _current_medoids_idx = best_medoids_idx

            possible_changes = itertools.product(_current_medoids_idx, _non_medoids_idx)
            change_happened = False
            for m, nm in possible_changes:

                _medoids_idx = list(_current_medoids_idx) \
                    .append(nm) \
                    .remove(m)

                _cost = self._calculate_cost(train_df, _medoids_idx)

                if best_cost > _cost:
                    best_cost = _cost
                    best_medoids_idx = _medoids_idx
                    change_happened = True

            if not change_happened:
                break

            if iteration >= self.max_iter:
                break

            print(f'iteration {iteration} finished')
            iteration += 1

        print(f'Finished with best cost {best_cost}. Final medoids:')
        print(self.final_medoids)
        self.final_medoids = train_df.loc[best_medoids_idx]

    def _find_cluster(self, medoids, row):
        distances = medoids.apply(lambda x: self.calc_distance(x.values, row.values, axis=0), axis=1)
        medoid_idx = int(distances.idxmin())

        return np.insert(distances.values, 0, medoid_idx)

    def predict(self, df: pd.DataFrame):
        if self.final_medoids is None:
            raise Exception('fit must be called before predict')

        results = df.apply(lambda row: self._find_cluster(self.final_medoids, row), axis=1)
        return results.apply(lambda x: x[0])
