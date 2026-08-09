import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

def simulate_ab_test(n_a, n_b, p_a, p_b):
    np.random.seed(42)
    conversions_a = np.random.binomial(n_a, p_a)
    conversions_b = np.random.binomial(n_b, p_b)
    return conversions_a, conversions_b

def compute_z_test(conversions_a, n_a, conversions_b, n_b):
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    p_pool = (conversions_a + conversions_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

def compute_confidence_interval(conversions, n, confidence):
    p = conversions / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se = np.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se

def power_analysis(effect_size, alpha, power):
    n = sms.NormalIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1)
    return n

def cuped_estimator(y_post, y_pre, x_post, x_pre):
    cov = np.cov(y_post, y_pre)[0, 1]
    var = np.var(y_pre)
    theta = cov / var
    y_cuped = y_post - theta * (y_pre - np.mean(y_pre))
    x_cuped = x_post - theta * (x_pre - np.mean(x_pre))
    return y_cuped, x_cuped

def run_simulation_study(n_simulations, n_per_group, true_effect, alpha):
    np.random.seed(42)
    rejections = 0
    p_a = 0.1
    p_b = p_a + true_effect
    for _ in range(n_simulations):
        conversions_a = np.random.binomial(n_per_group, p_a)
        conversions_b = np.random.binomial(n_per_group, p_b)
        _, p_val = compute_z_test(conversions_a, n_per_group, conversions_b, n_per_group)
        if p_val < alpha:
            rejections += 1
    return rejections / n_simulations

def plot_conversion_comparison(results): pass
def plot_power_curve(effect_sizes, sample_sizes): pass
def plot_cuped_variance_reduction(standard_var, cuped_var): pass
