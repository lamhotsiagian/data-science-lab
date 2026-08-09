import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
from customer_segmentation import generate_customer_data, find_optimal_k, run_kmeans, profile_segments
import numpy as np

def test_customer_segmentation():
    df = generate_customer_data(100)
    inertias, silhouettes = find_optimal_k(df, range(1, 6))
    
    assert max(silhouettes) > 0.3
    
    model, labels = run_kmeans(df, 4)
    profiles = profile_segments(df, labels)
    
    assert len(profiles) == 4
