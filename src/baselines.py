"""Simple reference forecasts that every ML model should beat."""

import numpy as np
import pandas as pd


def persistence_forecast(df, train, test, lag_days=7):
    """'Same hour last week' forecast for every test row.

    The value is looked up by timestamp in the full recorded series (past data
    only, so nothing from the future is used). The hourly timeline has gaps
    (165 missing hours), so a few lag values do not exist; for those rows the
    forecast falls back to the training-period mean for the same hour and
    day type. Returns a numpy array aligned with `test`.
    """
    ts = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')
    series = pd.Series(df['cnt'].values, index=ts)

    test_ts = test['dteday'] + pd.to_timedelta(test['hr'], unit='h')
    lagged = series.reindex(test_ts - pd.Timedelta(days=lag_days)).values

    fallback = train.groupby(['hr', 'workingday'])['cnt'].mean()
    keys = list(zip(test['hr'], test['workingday']))
    fill = np.array([fallback.get(k, train['cnt'].mean()) for k in keys])
    return np.where(np.isnan(lagged), fill, lagged)
