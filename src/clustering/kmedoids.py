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

    def _calculate_distances_to_medoids(self, df, medoids_idx):
        np_df = df.to_numpy()

        not_medoids = np.delete(df.to_numpy(), medoids_idx, axis=0)
        medoids = np_df[medoids_idx]

        cost_table = np.concatenate([self.calc_distance(not_medoids, m, axis=1) for m in medoids]) \
            .reshape(len(medoids), len(not_medoids))
        return cost_table

    def _calculate_cost(self, train_df, medoids_idx):
        cost_table = self._calculate_distances_to_medoids(train_df, medoids_idx)
        return np.min(cost_table, axis=1).sum(-1)

    @staticmethod
    def _get_possible_changes(medoids_idx, non_medoids_idx):
        possible_changes = itertools.product(medoids_idx, non_medoids_idx)

        def new_idx_list(m, nm):
            res = list(medoids_idx)
            res.append(nm)
            res.remove(m)
            return res

        return (new_idx_list(m, nm) for m, nm in possible_changes)

    def fit(self, train_df: pd.DataFrame):
        best_medoids_idx = self._initialize_random_medoids(train_df)
        best_cost = self._calculate_cost(train_df, best_medoids_idx)

        iteration = 1
        while True:
            _non_medoids_idx = np.delete(train_df.index.to_numpy(), best_medoids_idx)
            _current_medoids_idx = best_medoids_idx

            change_happened = False
            for _medoids_idx in KMedoids._get_possible_changes(_current_medoids_idx, _non_medoids_idx):

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

        self.final_medoids = train_df.loc[best_medoids_idx]
        print(f'Finished with best cost {best_cost}.')
        print('Final medoids:')
        print(self.final_medoids)

    def _assign_to_cluster(self, row):
        distances = self.final_medoids.apply(lambda x: self.calc_distance(x.values, row.values, axis=0), axis=1)
        medoid_idx = int(distances.idxmin())

        return np.insert(distances.values, 0, medoid_idx)

    @staticmethod
    def _map_labels_to_nice_number(labels):
        mapping = {original: new for original, new in zip(labels.unique(), itertools.count())}
        return labels.map(mapping)

    def predict(self, df: pd.DataFrame):
        if self.final_medoids is None:
            raise Exception('fit must be called before predict')

        results = df.apply(lambda row: self._assign_to_cluster(row), axis=1)
        labels = results.apply(lambda x: x[0])

        return self._map_labels_to_nice_number(labels)
