import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pytest
import numpy as np
from diagnostics import *

def test_diagnostics():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = 2 + 3*X[:,0] + np.random.randn(100)
    
    result = fit_ols_model(X, y)
    summ = extract_summary_components(result)
    assert "rsquared" in summ
    
    p_norm = check_normality(result.resid)
    assert 0 <= p_norm <= 1
    
    p_het = check_heteroscedasticity(result)
    assert 0 <= p_het <= 1
    
    dw = check_autocorrelation(result)
    assert 0 <= dw <= 4
    
    report = full_diagnostic_report(X, y)
    assert "summary" in report
    
    res_diag = compute_residual_diagnostics(result)
    assert "residuals" in res_diag
