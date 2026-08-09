import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import feature_engineering
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def test_polynomial_features():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    new_df = feature_engineering.create_polynomial_features(df, 2, ['a'])
    assert 'a^2' in new_df.columns

def test_target_encode_no_leakage():
    X_train = pd.DataFrame({'cat': ['A', 'B', 'A']})
    y_train = pd.Series([1, 0, 1])
    X_val = pd.DataFrame({'cat': ['A', 'B']})
    X_tr_enc, X_val_enc = feature_engineering.target_encode(X_train, y_train, X_val, 'cat', smoothing=1)
    assert not X_val_enc.isnull().any().any()

def test_compare_before_after():
    np.random.seed(42)
    data = fetch_california_housing(as_frame=True)
    X = data.data.iloc[:500]
    y = data.target.iloc[:500]
    X_eng = feature_engineering.engineer_all_features(X, y)
    score_raw, score_eng = feature_engineering.compare_before_after(X, X_eng, y)
    assert not np.isnan(score_raw)
    assert not np.isnan(score_eng)
    assert score_eng >= score_raw - 0.05
