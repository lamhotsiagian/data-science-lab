import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pca_tsne import load_digits_data, fit_pca, compute_cumulative_variance, find_optimal_components, fit_tsne

def test_pca_tsne():
    X, y = load_digits_data()
    X = X[:100] 
    y = y[:100]
    
    pca, X_pca = fit_pca(X, n_components=10)
    cumulative = compute_cumulative_variance(pca)
    
    assert cumulative[-1] <= 1.0001
    assert np.all(np.diff(cumulative) >= 0)
    
    X_tsne = fit_tsne(X, n_components=2, perplexity=10)
    assert X_tsne.shape == (100, 2)
