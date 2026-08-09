import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from category_encoders import TargetEncoder

def create_polynomial_features(X, degree, columns):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(X[columns])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(columns), index=X.index)
    X_new = pd.concat([X.drop(columns, axis=1), poly_df], axis=1)
    return X_new

def create_interaction_features(X, col_pairs):
    X_new = X.copy()
    for col1, col2 in col_pairs:
        X_new[f"{col1}_{col2}_interact"] = X[col1] * X[col2]
    return X_new

def bin_continuous(X, col, n_bins, strategy):
    X_new = X.copy()
    if strategy == 'uniform':
        X_new[f"{col}_binned"] = pd.cut(X[col], bins=n_bins, labels=False)
    elif strategy == 'quantile':
        X_new[f"{col}_binned"] = pd.qcut(X[col], q=n_bins, labels=False, duplicates='drop')
    return X_new

def target_encode(X_train, y_train, X_val, col, smoothing):
    encoder = TargetEncoder(cols=[col], smoothing=smoothing)
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_train_enc[col] = encoder.fit_transform(X_train[col], y_train)
    X_val_enc[col] = encoder.transform(X_val[col])
    return X_train_enc, X_val_enc

def one_hot_encode(X, cols):
    X_new = pd.get_dummies(X, columns=cols, drop_first=True)
    return X_new

def ordinal_encode(X, col, order):
    encoder = OrdinalEncoder(categories=[order])
    X_new = X.copy()
    X_new[col] = encoder.fit_transform(X[[col]])
    return X_new

def select_features_mutual_info(X, y, k):
    mi = mutual_info_regression(X, y, random_state=42)
    top_k_indices = np.argsort(mi)[-k:]
    return X.columns[top_k_indices].tolist()

def engineer_all_features(X, y):
    X_new = create_interaction_features(X, [('MedInc', 'HouseAge')])
    X_new = bin_continuous(X_new, 'HouseAge', 5, 'quantile')
    return X_new

def compare_before_after(X_raw, X_engineered, y):
    score_raw = np.mean(cross_val_score(Ridge(random_state=42), X_raw, y, cv=3))
    score_eng = np.mean(cross_val_score(Ridge(random_state=42), X_engineered, y, cv=3))
    return score_raw, score_eng
