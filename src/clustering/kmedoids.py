from collections import OrderedDict
from typing import Callable

import pandas as pd
import numpy as np


class KMedoids:

    def __init__(self,
                 n_clusters: int,
                 distance_metric: Callable[[pd.Series, pd.Series], float],
                 random_state: int = None):
        self.k = n_clusters
        self.dst = distance_metric
        self.random_state = random_state

        self._df = None
        self.clusters = None
        self.medoids_idx = None

    def _initialize(self):
        index = self._df.index.to_series()
        medoids = index.sample(self.k, random_state=self.random_state)

        self.medoids_idx = medoids.to_list()

    def _get_medoids(self):
        return self._df[self._df.index.isin(self.medoids_idx)]

    def _get_non_medoids(self):
        return self._df[~self._df.index.isin(self.medoids_idx)]

    def _assign_to_clusters(self):
        not_medoids = self._get_non_medoids()
        medoids = self._get_medoids()

        def find_cluster(row):
            distances = medoids.apply(lambda x: self.dst(x, row), axis=1)
            medoid_idx = int(distances.idxmin())

            return [str(medoid_idx)] + distances.to_list()

        clusters = not_medoids.apply(find_cluster, axis=1)
        self.clusters = pd.DataFrame.from_dict(OrderedDict(clusters)) \
            .transpose()
        self.clusters.columns = ['medoid'] + [str(i) for i in medoids.index.to_list()]

    def _calculate_cost(self) -> float:
        not_medoids = self._get_non_medoids()

        def distance_to_cluster(row) -> float:
            idx = row.name
            r_cluster = self.clusters.loc[idx]

            return r_cluster[r_cluster['medoid']]

        all_costs = not_medoids.apply(distance_to_cluster, axis=1)
        return all_costs.sum()

    def fit(self, df: pd.DataFrame):
        self._df = df

        self._initialize()
        self._assign_to_clusters()
        print(self._calculate_cost())

        pass

    def predict(self, df: pd.DataFrame):
        pass


if __name__ == '__main__':
    from src.distance import euclidean_distance
    import pandas as pd

    km = KMedoids(n_clusters=3, distance_metric=euclidean_distance, random_state=123)
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'e': [10, 20, 30, 40, 50]})

    km.fit(df)
