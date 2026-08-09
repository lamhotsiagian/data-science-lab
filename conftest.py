"""
Shared pytest fixtures for the Data Science Lab curriculum.
Provides reusable test utilities across all 14 chapters.
"""

import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="dslab_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def project_root():
    """Return the absolute path to the data-science-lab root."""
    return os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Sample Data Generators
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_numeric_df():
    """A small numeric DataFrame for quick tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, 100),
        "feature_b": rng.normal(5, 2, 100),
        "feature_c": rng.uniform(0, 10, 100),
        "target": rng.normal(10, 3, 100),
    })


@pytest.fixture
def sample_mixed_df():
    """A DataFrame with mixed types including categoricals and nulls."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "id": range(n),
        "name": [f"item_{i}" for i in range(n)],
        "category": rng.choice(["A", "B", "C", "D"], n),
        "value": rng.normal(50, 15, n),
        "count": rng.integers(0, 100, n),
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    })
    # Inject some nulls
    mask = rng.random(n) < 0.1
    df.loc[mask, "value"] = np.nan
    mask2 = rng.random(n) < 0.05
    df.loc[mask2, "category"] = None
    return df


@pytest.fixture
def sample_classification_data():
    """Generate a simple binary classification dataset."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def sample_regression_data():
    """Generate a simple regression dataset."""
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=3,
        noise=10.0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def sample_csv_path(tmp_dir):
    """Create a sample CSV file and return its path."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "id": range(50),
        "value": rng.normal(100, 25, 50),
        "category": rng.choice(["X", "Y", "Z"], 50),
    })
    path = os.path.join(tmp_dir, "sample.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_close():
    """Helper fixture for floating point comparison."""
    def _assert_close(a, b, rtol=1e-5, atol=1e-8):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    return _assert_close
