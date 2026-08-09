import pandas as pd
import numpy as np

def load_and_validate(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")
    return df.dropna().copy()

def compute_summary_stats(df):
    return df.describe().to_dict()

def filter_dataframe(df, filters_dict):
    filtered = df.copy()
    for col, val in filters_dict.items():
        if isinstance(val, (list, tuple)):
            filtered = filtered[filtered[col].isin(val)]
        else:
            filtered = filtered[filtered[col] == val]
    return filtered

def compute_distribution(df, col, bins=10):
    counts, edges = np.histogram(df[col], bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}

def compute_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()

def compute_grouped_aggregation(df, group_col, agg_col, func="mean"):
    return df.groupby(group_col)[agg_col].agg(func).reset_index()
