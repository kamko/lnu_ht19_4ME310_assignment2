import pandas as pd
import numpy as np


def euclidean_distance(v1: pd.Series, v2: pd.Series) -> float:
    _assert_equal_sizes(v1, v2)

    res = np.sqrt(euclidean_squared_distance(v1, v2))
    return res


def euclidean_squared_distance(v1: pd.Series, v2: pd.Series) -> float:
    _assert_equal_sizes(v1, v2)

    res = v1 - v2
    res = res ** 2
    return res.sum()


def manhattan_distance(v1: pd.Series, v2: pd.Series) -> float:
    _assert_equal_sizes(v1, v2)

    pass


def _assert_equal_sizes(v1: pd.Series, v2: pd.Series):
    if len(v1) != len(v2):
        raise ValueError(f'Invalid input len(v1)={len(v1)} and len(v2)={len(v2)}')
