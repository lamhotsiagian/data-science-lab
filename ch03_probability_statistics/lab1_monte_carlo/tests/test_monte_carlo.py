import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import monte_carlo
import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.stats as stats

def test_estimate_pi():
    pi_est = monte_carlo.estimate_pi(100000)
    assert abs(pi_est - np.pi) < 0.05

def test_option_price():
    mc_price = monte_carlo.option_price_monte_carlo(100, 100, 0.05, 0.2, 1, 100000)
    bs_price = monte_carlo.black_scholes_call(100, 100, 0.05, 0.2, 1)
    assert abs(mc_price - bs_price) / bs_price < 0.05

def test_clt():
    results = monte_carlo.coin_flip_clt(100, 1000)
    assert len(results) == 1000
