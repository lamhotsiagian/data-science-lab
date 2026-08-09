import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import time_series_features
import pandas as pd
import numpy as np

def test_rolling_features():
    df = pd.DataFrame({'val': np.arange(10)})
    res = time_series_features.create_rolling_features(df, 'val', [3])
    assert 'val_roll_mean_3' in res.columns
    assert np.isnan(res['val_roll_mean_3'].iloc[0])

def test_adf_test():
    series = time_series_features.generate_synthetic_timeseries(100, 0, 10, 1)
    stat, pval, is_stat = time_series_features.adf_test(series)
    assert isinstance(is_stat, bool)

def test_decomposition():
    df = pd.DataFrame({'val': time_series_features.generate_synthetic_timeseries(100, 0, 10, 1)})
    trend, seasonal, resid = time_series_features.seasonal_decompose_stl(df, 'val', 10)
    assert np.allclose(df['val'], trend + seasonal + resid)
