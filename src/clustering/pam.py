import itertools

import numpy as np

from src.log_util import log
from src.numpy_util import np_rows
from src.numpy_util import random_from_range
from src.numpy_util import to_numpy


class PAM:

    def __init__(self,
                 n_clusters,
                 distance,
                 max_iter=100,
                 random_state=None,
                 verbose=True):
        self.k = n_clusters
        self.calc_distance = distance
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self.final_medoids = None

    def _get_random_medoids(self, data):
        return random_from_range(self.k, np_rows(data), self.random_state)

    def _calculate_distances_to_medoids(self, data, medoids):
        return np.concatenate([self.calc_distance(data, m, axis=1) for m in medoids]) \
            .reshape(len(medoids), len(data))

    def _calculate_cost(self, data, medoids_idx):
        not_medoids = np.delete(data, medoids_idx, axis=0)
        medoids = data[medoids_idx]

        cost_table = self._calculate_distances_to_medoids(not_medoids, medoids)
        return np.min(cost_table, axis=0).sum(-1)

    @staticmethod
    def _get_possible_changes(medoids_idx, non_medoids_idx):
        possible_changes = itertools.product(medoids_idx, non_medoids_idx)

        def new_idx_list(m, nm):
            res = list(medoids_idx)
            res.append(nm)
            res.remove(m)
            return res

        return (new_idx_list(m, nm) for m, nm in possible_changes)

    def fit(self, data):
        data = to_numpy(data)

        best_medoids_idx = self._get_random_medoids(data)
        best_cost = self._calculate_cost(data, best_medoids_idx)

        whole_index = np.arange(np_rows(data))

        iteration = 1
        while True:
            _non_medoids_idx = np.delete(whole_index, best_medoids_idx)
            _current_medoids_idx = best_medoids_idx

            change_happened = False
            for _medoids_idx in PAM._get_possible_changes(_current_medoids_idx, _non_medoids_idx):

                _cost = self._calculate_cost(data, _medoids_idx)
                if best_cost > _cost:
                    best_cost = _cost
                    best_medoids_idx = _medoids_idx
                    change_happened = True

            if not change_happened:
                break

            if iteration >= self.max_iter:
                break

            log(f'iteration {iteration} finished', self.verbose)
            iteration += 1

        self.final_medoids = data[best_medoids_idx]
        log(f'Finished with best cost {best_cost}.', self.verbose)
        log('Final medoids:', self.verbose)
        log(self.final_medoids, self.verbose)

    def predict(self, data):
        if self.final_medoids is None:
            raise Exception('fit must be called before predict')

        distances = self._calculate_distances_to_medoids(data, self.final_medoids)
        return np.argmin(distances, axis=0)
