import pandas as pd


def min_max(v: pd.Series) -> pd.Series:
    vmin = v.min()
    vmax = v.max()

    def action(value): return (value - vmin) / (vmax - vmin)

    return v.apply(action)
