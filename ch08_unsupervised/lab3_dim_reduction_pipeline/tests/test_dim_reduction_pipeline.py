import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from dim_reduction_pipeline import DimReductionPipeline, compare_methods
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

def test_pipeline():
    X, y = make_blobs(n_samples=100, centers=3, n_features=10, random_state=42)
    
    pipeline = DimReductionPipeline()
    X_reduced, labels, metrics = pipeline.fit_transform_cluster(X, n_components=2, cluster_method='kmeans', n_clusters=3)
    
    assert X_reduced.shape == (100, 2)
    assert len(np.unique(labels)) == 3
    assert 'silhouette' in metrics
    
    methods = {'PCA': PCA(n_components=2, random_state=42)}
    comp_results = compare_methods(X, y, methods, 2)
    assert 'PCA' in comp_results
