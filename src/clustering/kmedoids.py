from collections import OrderedDict
from typing import Callable

import pandas as pd

import itertools


class KMedoids:

    def __init__(self,
                 n_clusters: int,
                 distance_metric: Callable[[pd.Series, pd.Series], float],
                 max_iter: int = 100,
                 random_state: int = None):
        self.k = n_clusters
        self.calc_distance = distance_metric
        self.max_iter = max_iter
        self.random_state = random_state

        self.final_medoids = None

    def _initialize_random_medoids(self, df):
        index = df.index.to_series()
        medoids = index.sample(self.k, random_state=self.random_state)

        return set(medoids.to_list())

    @staticmethod
    def _get_medoids(df, medoids_idx):
        return df[df.index.isin(medoids_idx)]

    @staticmethod
    def _get_non_medoids(df, medoids_idx):
        return df[~df.index.isin(medoids_idx)]

    def _find_cluster(self, medoids, row):
        distances = medoids.apply(lambda x: self.calc_distance(x, row), axis=1)
        medoid_idx = int(distances.idxmin())

        return [str(medoid_idx)] + distances.to_list()

    def _assign_to_clusters(self, df, medoids_idx):
        not_medoids = self._get_non_medoids(df, medoids_idx)
        medoids = self._get_medoids(df, medoids_idx)

        clusters = not_medoids.apply(lambda row: self._find_cluster(medoids, row), axis=1)
        df_clusters = pd.DataFrame.from_dict(OrderedDict(clusters)) \
            .transpose()
        df_clusters.columns = ['medoid'] + [str(i) for i in medoids.index.to_list()]

        return df_clusters

    def _calculate_cost(self, df, df_clusters, medoids_idx) -> float:
        not_medoids = self._get_non_medoids(df, medoids_idx)

        def distance_to_cluster(row) -> float:
            idx = row.name
            r_cluster = df_clusters.loc[idx]

            return r_cluster[r_cluster['medoid']]

        all_costs = not_medoids.apply(distance_to_cluster, axis=1)
        return all_costs.sum()

    def fit(self, train_df: pd.DataFrame):
        medoids_idx = self._initialize_random_medoids(train_df)
        clusters = self._assign_to_clusters(train_df, medoids_idx)
        current_cost = self._calculate_cost(train_df, clusters, medoids_idx)

        best_medoids_idx = medoids_idx
        best_cost = current_cost

        iteration = 0
        while True:
            _non_medoids_idx = self._get_non_medoids(train_df, best_medoids_idx).index
            possible_changes = itertools.product(best_medoids_idx, _non_medoids_idx)

            change_happened = False
            for m, nm in possible_changes:
                _medoids_idx = best_medoids_idx.copy()

                _medoids_idx.add(nm)
                _medoids_idx.remove(m)

                _clusters = self._assign_to_clusters(train_df, _medoids_idx)
                _cost = self._calculate_cost(train_df, _clusters, _medoids_idx)

                if best_cost > _cost:
                    best_cost = _cost
                    best_medoids_idx = _medoids_idx

            if not change_happened:
                break

            if iteration > self.max_iter:
                break

            iteration += 1

        self.final_medoids = train_df.loc[best_medoids_idx]

    def predict(self, df: pd.DataFrame):
        if self.final_medoids is None:
            raise Exception('fit must be called before predict')

        results = df.apply(lambda row: self._find_cluster(self.final_medoids, row), axis=1)
        return results.apply(lambda x: x[0])


if __name__ == '__main__':
    from src.distance import euclidean_distance
    import pandas as pd

    km = KMedoids(n_clusters=3, distance_metric=euclidean_distance, random_state=123)
    dfx = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'e': [10, 20, 30, 40, 50]})

    km.fit(dfx)
    r = km.predict(dfx)

    print(r)
