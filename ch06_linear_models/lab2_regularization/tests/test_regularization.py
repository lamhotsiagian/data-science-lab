import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pytest
import numpy as np
from sklearn.linear_model import Lasso
from regularization import *

def test_regularization():
    X, y = generate_multicollinear_data(100, 5, 3, 0.9)
    res = fit_all_models(X, y, X, y, [0.1, 1.0])
    assert 0.1 in res
    
    paths = compute_coefficient_paths(X, y, [0.1, 1.0], 'lasso')
    assert paths.shape == (2, 5)
    
    m = Lasso(alpha=100.0).fit(X, y)
    assert np.sum(m.coef_ == 0) > 0
    
    cv_res = compare_models_cv(X, y)
    assert "Ridge" in cv_res
