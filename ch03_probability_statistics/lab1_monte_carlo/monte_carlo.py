import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def estimate_pi(n_samples):
    np.random.seed(42)
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    inside = (x**2 + y**2) <= 1
    return 4 * np.sum(inside) / n_samples

def coin_flip_clt(n_flips, n_experiments):
    np.random.seed(42)
    flips = np.random.binomial(n_flips, 0.5, n_experiments)
    return flips

def option_price_monte_carlo(S0, K, r, sigma, T, n_simulations):
    np.random.seed(42)
    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoffs)

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def plot_pi_convergence(max_samples):
    samples = np.logspace(2, np.log10(max_samples), 50).astype(int)
    estimates = [estimate_pi(n) for n in samples]
    plt.plot(samples, estimates)

def plot_clt_demonstration(n_values):
    plt.hist(n_values, bins=30, density=True)

def plot_option_distribution(prices):
    plt.hist(prices, bins=50, density=True)
