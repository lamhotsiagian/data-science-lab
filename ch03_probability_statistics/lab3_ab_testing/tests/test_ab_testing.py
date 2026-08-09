import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import ab_testing
import matplotlib
matplotlib.use('Agg')
import numpy as np

def test_type_I_error():
    error_rate = ab_testing.run_simulation_study(1000, 1000, 0, 0.05)
    assert 0.03 <= error_rate <= 0.07

def test_power_increases_with_n():
    n1 = ab_testing.power_analysis(0.1, 0.05, 0.8)
    n2 = ab_testing.power_analysis(0.1, 0.05, 0.9)
    assert n2 > n1

def test_cuped_variance_reduction():
    np.random.seed(42)
    y_pre = np.random.normal(0, 1, 1000)
    y_post = y_pre + np.random.normal(0, 0.5, 1000)
    x_pre = np.random.normal(0, 1, 1000)
    x_post = x_pre + np.random.normal(0, 0.5, 1000)
    
    y_cuped, x_cuped = ab_testing.cuped_estimator(y_post, y_pre, x_post, x_pre)
    assert np.var(y_cuped) < np.var(y_post)

def test_z_test():
    z, p = ab_testing.compute_z_test(100, 1000, 150, 1000)
    assert p < 0.05
