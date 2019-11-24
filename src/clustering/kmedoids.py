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

        return set(medoids.to_list())

    @staticmethod
    def _get_medoids(df, medoids_idx):
        return df[df.index.isin(medoids_idx)]

    @staticmethod
    def _get_non_medoids(df, medoids_idx):
        return df[~df.index.isin(medoids_idx)]

    def _find_cluster(self, medoids, row):
        distances = medoids.apply(lambda x: self.calc_distance(x.values, row.values, axis=0), axis=1)
        medoid_idx = int(distances.idxmin())

        return np.insert(distances.values, 0, medoid_idx)

    def _assign_to_clusters(self, df, medoids_idx):
        not_medoids = self._get_non_medoids(df, medoids_idx)
        medoids = self._get_medoids(df, medoids_idx)

        res = []
        for i in medoids.values:
            res.append(self.calc_distance(not_medoids, i, axis=1))
        return pd.DataFrame(res)

    @staticmethod
    def _calculate_cost(df_clusters):
        return df_clusters.min(axis=1).sum()

    def fit(self, train_df: pd.DataFrame):
        medoids_idx = self._initialize_random_medoids(train_df)
        clusters = self._assign_to_clusters(train_df, medoids_idx)
        current_cost = self._calculate_cost(clusters)

        best_medoids_idx = medoids_idx
        best_cost = current_cost

        iteration = 1
        while True:
            _non_medoids_idx = self._get_non_medoids(train_df, best_medoids_idx).index

            _original_medoids_idx = best_medoids_idx
            possible_changes = itertools.product(_original_medoids_idx, _non_medoids_idx)

            change_happened = False
            for m, nm in possible_changes:
                _medoids_idx = _original_medoids_idx.copy()

                _medoids_idx.add(nm)
                _medoids_idx.remove(m)

                _clusters = self._assign_to_clusters(train_df, _medoids_idx)
                _cost = self._calculate_cost(_clusters)

                # print(best_cost, _cost)
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

    def predict(self, df: pd.DataFrame):
        if self.final_medoids is None:
            raise Exception('fit must be called before predict')

        results = df.apply(lambda row: self._find_cluster(self.final_medoids, row), axis=1)
        return results.apply(lambda x: x[0])


if __name__ == '__main__':
    from src.distance import euclidean_squared_distance
    import pandas as pd

    km = KMedoids(n_clusters=3, distance_metric=euclidean_squared_distance, random_state=123)
    dfx = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'e': [10, 20, 30, 40, 50]})

    km.fit(dfx)
    r = km.predict(dfx)

    print(r)
