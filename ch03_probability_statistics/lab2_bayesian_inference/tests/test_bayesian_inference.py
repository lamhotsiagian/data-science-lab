import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import bayesian_inference
import matplotlib
matplotlib.use('Agg')
import numpy as np

def test_grid_posterior_sum():
    grid = np.linspace(0, 1, 100)
    prior = np.ones(100) / 100
    likelihood_func = lambda g, d: g**d * (1-g)**(1-d)
    posterior = bayesian_inference.grid_posterior(prior, likelihood_func, 1, grid)
    assert np.isclose(np.sum(posterior), 1.0)

def test_medical_test():
    prob = bayesian_inference.medical_test_bayes(0.01, 0.99, 0.99)
    assert np.isclose(prob, 0.5)

def test_beta_binomial_update():
    new_alpha, new_beta = bayesian_inference.beta_binomial_update(1, 1, 10, 5)
    assert new_alpha == 11
    assert new_beta == 6
