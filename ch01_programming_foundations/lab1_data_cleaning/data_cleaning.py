import pandas as pd
import polars as pl
import numpy as np

def load_messy_csv(path):
    return pd.read_csv(path)

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def standardize_dates(df, col):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def clean_numeric(df, col):
    df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def handle_missing(df, strategy='median'):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
    return df

def standardize_categories(df, col, mapping):
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().map(mapping).fillna(df[col])
    return df

def validate_schema(df, expected_dtypes):
    for col, dtype in expected_dtypes.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except ValueError:
                pass
    return df

def clean_pipeline(path):
    df = load_messy_csv(path)
    df = clean_column_names(df)
    df = standardize_dates(df, 'join_date')
    df = clean_numeric(df, 'salary')
    df = clean_numeric(df, 'age')
    df = remove_duplicates(df)
    df = handle_missing(df, strategy='median')
    gender_map = {'m': 'male', 'f': 'female', 'male': 'male', 'female': 'female'}
    df = standardize_categories(df, 'gender', gender_map)
    return df

def clean_pipeline_polars(path):
    df = pl.read_csv(path, infer_schema_length=0) # Read all as string to handle mess
    df = df.rename({c: c.strip().lower().replace(' ', '_') for c in df.columns})
    
    df = df.with_columns([
        pl.col('age').str.replace_all(r'[^0-9]', '', literal=False).cast(pl.Int64, strict=False),
        pl.col('salary').str.replace_all(r'[^0-9]', '', literal=False).cast(pl.Int64, strict=False),
    ])
    
    df = df.unique()
    return df
