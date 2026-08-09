import numpy as np
from sklearn.datasets import make_classification
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def generate_reference_data(n_samples=1000, n_features=5):
    X, _ = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    return X

def simulate_drift(reference_data, drift_type="mean_shift", magnitude=1.0):
    drifted = reference_data.copy()
    if drift_type == "mean_shift":
        drifted += magnitude
    elif drift_type == "variance_change":
        drifted *= magnitude
    elif drift_type == "feature_swap" and reference_data.shape[1] >= 2:
        drifted[:, 0], drifted[:, 1] = drifted[:, 1], drifted[:, 0].copy()
    elif drift_type == "gradual":
        n = len(drifted)
        factors = np.linspace(0, magnitude, n).reshape(-1, 1)
        drifted += factors
    return drifted

def ks_test_per_feature(reference, production):
    n_features = reference.shape[1]
    stats, pvals = [], []
    for i in range(n_features):
        stat, pval = ks_2samp(reference[:, i], production[:, i])
        stats.append(stat)
        pvals.append(pval)
    return stats, pvals

def detect_univariate_drift(reference, production, alpha=0.05):
    _, pvals = ks_test_per_feature(reference, production)
    return [p < alpha for p in pvals]

class AutoencoderDriftDetector:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(4, 2, 4), max_iter=500, random_state=42)
        self.threshold = None
        
    def fit(self, reference_data):
        self.model.fit(reference_data, reference_data)
        errors = self.compute_reconstruction_error(reference_data)
        self.threshold = np.percentile(errors, 95)
        
    def compute_reconstruction_error(self, data):
        preds = self.model.predict(data)
        return np.mean((data - preds)**2, axis=1)
        
    def detect_drift(self, production_data, threshold_percentile=None):
        if self.threshold is None:
            raise ValueError("Fit model first")
        errors = self.compute_reconstruction_error(production_data)
        thresh = np.percentile(errors, threshold_percentile) if threshold_percentile else self.threshold
        return errors > thresh

def run_drift_simulation(n_features=5, drift_magnitudes=[0.1, 0.5, 1.0]):
    ref = generate_reference_data(500, n_features)
    results = {}
    for mag in drift_magnitudes:
        prod = simulate_drift(ref, "mean_shift", mag)
        flags = detect_univariate_drift(ref, prod)
        results[mag] = sum(flags) / len(flags)
    return results

def plot_feature_distributions(reference, production, feature_idx=0):
    fig, ax = plt.subplots()
    ax.hist(reference[:, feature_idx], alpha=0.5, label='Reference')
    ax.hist(production[:, feature_idx], alpha=0.5, label='Production')
    ax.legend()
    return fig

def plot_drift_scores(ks_stats, feature_names=None):
    fig, ax = plt.subplots()
    feature_names = feature_names or [f"F{i}" for i in range(len(ks_stats))]
    ax.bar(feature_names, ks_stats)
    return fig

def plot_reconstruction_errors(ref_errors, prod_errors):
    fig, ax = plt.subplots()
    ax.hist(ref_errors, alpha=0.5, label='Reference')
    ax.hist(prod_errors, alpha=0.5, label='Production')
    ax.legend()
    return fig

def plot_drift_timeline(errors_over_time, threshold):
    fig, ax = plt.subplots()
    ax.plot(errors_over_time)
    ax.axhline(threshold, color='r', linestyle='--')
    return fig
