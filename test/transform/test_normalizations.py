import pytest
from pandas import Series
from pandas.testing import assert_series_equal

from src.transform import min_max


@pytest.mark.parametrize('input, expected', [
    (
            Series([0, 50, 100]),
            Series([0, 0.5, 1]),
    ),
    (
            Series([0, 1000, 10000]),
            Series([0, 0.1, 1]),
    ),
    (
            Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            Series([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
    ),
])
def test_min_max_transform(input, expected):
    res = min_max(input)

    assert_series_equal(res, expected)
