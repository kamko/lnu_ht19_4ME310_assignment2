import pytest
from pandas import Series
from numpy.testing import assert_almost_equal

from src.distance import euclidean_distance


@pytest.mark.parametrize('v1, v2, expected', [
    (
            Series([45, 0.05]),
            Series([60, 0.05]),
            15
    ),
    (
            Series([45, 0.05]),
            Series([60, 0.05]),
            15
    )
])
def test_euclidean_distance(v1, v2, expected):
    res = euclidean_distance(v1, v2)

    assert_almost_equal(res, expected)
