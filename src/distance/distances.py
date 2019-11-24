import pandas as pd
import numpy as np


def euclidean_distance(v1, v2, axis=0):
    return np.sqrt(euclidean_squared_distance(v1, v2, axis))


def euclidean_squared_distance(v1, v2, axis=0):
    return np.sum((v1 - v2) ** 2, axis=axis)


def manhattan_distance(v1, v2, axis=0):
    return np.sum(np.abs(v1 - v2), axis=axis)
