import pytest
from pandas import Series
from numpy.testing import assert_almost_equal

from src.distance import euclidean_distance

from src.distance import euclidean_squared_distance


@pytest.mark.parametrize('v1, v2, expected', [
    (
            Series([45, 0.05]),
            Series([60, 0.05]),
            15
    ),
    (
            Series([12.231, 13.231, 1524.1233, 1254.123, 12315.33]),
            Series([1211.231, 12, 0, -24, -15.25]),
            12547.404
    )
])
def test_euclidean_distance(v1, v2, expected):
    res = euclidean_distance(v1, v2)

    assert_almost_equal(res, expected, decimal=3)


@pytest.mark.parametrize('v1, v2, expected', [
    (
            Series([45, 0.05]),
            Series([60, 0.05]),
            225
    ),
    (
            Series([12.231, 13.231, 1524.1233, 1254.123, 12315.33]),
            Series([1211.231, 12, 0, -24, -15.25]),
            157437355.888
    )
])
def test_euclidean_squared_distance(v1, v2, expected):
    res = euclidean_squared_distance(v1, v2)

    assert_almost_equal(res, expected, decimal=3)
