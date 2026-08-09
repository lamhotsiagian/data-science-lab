import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from ols import *

def test_ols():
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = 2 + 3*X[:,0] + 4*X[:,1] + np.random.randn(100)
    
    comp = compare_with_sklearn(X, y)
    np.testing.assert_allclose(comp["custom_coef"], comp["sklearn_coef"], rtol=1e-5)
    
    model = OLSRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    sk = LinearRegression().fit(X, y)
    np.testing.assert_allclose(y_pred, sk.predict(X), rtol=1e-5)
    
    r2 = compute_r_squared(y, y_pred)
    assert np.isclose(r2, sk.score(X, y))
