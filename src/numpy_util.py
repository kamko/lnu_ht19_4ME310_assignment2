import numpy as np
import pandas as pd


def to_numpy(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy()

    if isinstance(obj, pd.Series):
        return obj.to_numpy()

    if isinstance(obj, np.ndarray):
        return obj

    return np.array(obj)


def np_rows(arr):
    return np.shape(arr)[0]


def np_cols(arr):
    return np.shape(arr)[1]


def random_from_range(count, size, random_state=None):
    random = np.random.RandomState(random_state)

    arr = np.arange(size)
    return random.choice(arr, size=count, replace=False)
