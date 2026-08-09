import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeTransformer(BaseEstimator, TransformerMixin):
    """Expand each datetime column into year/month/day integer features.

    `fit` records the input column names so `get_feature_names_out` can emit
    real names. Returning None from that method breaks ColumnTransformer's
    `verbose_feature_names_out` and any downstream `set_output(transform=...)`.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for col in X.columns:
            dt = pd.to_datetime(X[col])
            X_new[f"{col}_year"] = dt.dt.year
            X_new[f"{col}_month"] = dt.dt.month
            X_new[f"{col}_day"] = dt.dt.day
        return X_new

    def get_feature_names_out(self, input_features=None):
        cols = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(
            [f"{c}_{part}" for c in cols for part in ("year", "month", "day")],
            dtype=object,
        )

class PreprocessingConfig:
    def __init__(self, config_dict):
        self.numeric_cols = config_dict.get('numeric_cols', [])
        self.categorical_cols = config_dict.get('categorical_cols', [])
        self.datetime_cols = config_dict.get('datetime_cols', [])

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            return cls(yaml.safe_load(f))

def build_pipeline(config):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
    datetime_transformer = Pipeline(steps=[
        ('dt', DatetimeTransformer())])
        
    transformers = []
    if config.numeric_cols:
        transformers.append(('num', numeric_transformer, config.numeric_cols))
    if config.categorical_cols:
        transformers.append(('cat', categorical_transformer, config.categorical_cols))
    if config.datetime_cols:
        transformers.append(('dt', datetime_transformer, config.datetime_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor)])

def detect_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return {'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols, 'datetime_cols': datetime_cols}

def create_default_config(df):
    types = detect_column_types(df)
    return PreprocessingConfig(types)

def fit_transform_pipeline(df, config):
    pipeline = build_pipeline(config)
    return pipeline.fit_transform(df), pipeline

def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)

def load_pipeline(path):
    return joblib.load(path)
