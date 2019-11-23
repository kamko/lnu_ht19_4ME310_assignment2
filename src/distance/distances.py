import pandas as pd
import numpy as np


def euclidean_distance(v1: pd.Series, v2: pd.Series) -> float:
    res = v1 - v2
    res = res ** 2
    res = np.sqrt(res.sum())

    return res


def manhattan_distance(v1: pd.Series, v2: pd.Series) -> float:
    pass


def euclidean_squared_distance(v1: pd.Series, v2: pd.Series) -> float:
    pass
