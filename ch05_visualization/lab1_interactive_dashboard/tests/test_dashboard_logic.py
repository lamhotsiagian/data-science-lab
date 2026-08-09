import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pytest
import pandas as pd
from dashboard_logic import *

def test_load_and_validate():
    df = pd.DataFrame({"a": [1, None, 3]})
    val = load_and_validate(df)
    assert len(val) == 2

def test_summary_stats():
    df = pd.DataFrame({"a": [1, 2, 3]})
    stats = compute_summary_stats(df)
    assert "a" in stats
    assert stats["a"]["mean"] == 2.0

def test_filter():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    f = filter_dataframe(df, {"b": ["x", "y"]})
    assert len(f) == 2

def test_dist():
    df = pd.DataFrame({"a": [1, 2, 3]})
    d = compute_distribution(df, "a", 2)
    assert len(d["counts"]) == 2

def test_corr():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    c = compute_correlation_matrix(df)
    assert c.loc["a", "b"] == 1.0

def test_grouped():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 3, 2]})
    g = compute_grouped_aggregation(df, "g", "v")
    assert g[g["g"] == "a"]["v"].iloc[0] == 2.0
