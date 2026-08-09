import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

def generate_synthetic_timeseries(n, trend, seasonality_period, noise_std):
    np.random.seed(42)
    t = np.arange(n)
    series = trend * t + 10 * np.sin(2 * np.pi * t / seasonality_period) + np.random.normal(0, noise_std, n)
    return pd.Series(series)

def create_rolling_features(df, col, windows):
    df_new = df.copy()
    for w in windows:
        df_new[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
        df_new[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
        df_new[f"{col}_roll_min_{w}"] = df[col].rolling(w).min()
        df_new[f"{col}_roll_max_{w}"] = df[col].rolling(w).max()
    return df_new

def create_lag_features(df, col, lags):
    df_new = df.copy()
    for lag in lags:
        df_new[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df_new

def seasonal_decompose_stl(df, col, period):
    stl = STL(df[col], period=period)
    res = stl.fit()
    return res.trend, res.seasonal, res.resid

def adf_test(series):
    result = adfuller(series.dropna())
    return result[0], result[1], bool(result[1] < 0.05)

def difference_series(series, d):
    return series.diff(periods=d)

def extract_all_features(df, col, windows, lags, period):
    df_new = create_rolling_features(df, col, windows)
    df_new = create_lag_features(df_new, col, lags)
    trend, seasonal, resid = seasonal_decompose_stl(df, col, period)
    df_new['trend'] = trend
    df_new['seasonal'] = seasonal
    df_new['resid'] = resid
    return df_new

def plot_decomposition(trend, seasonal, residual): pass
def plot_rolling_statistics(df, col, window): pass
