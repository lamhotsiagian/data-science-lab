import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import matplotlib
matplotlib.use('Agg')
from training_dynamics import generate_labeled_data, compare_orderings, compare_batch_sizes

def test_orderings():
    X, y = generate_labeled_data(200, 10, 2)
    l_sort, l_shuf = compare_orderings(X, y, epochs=10, batch_size=16, lr=0.1)
    # Shuffled usually converges better/faster
    assert l_shuf[-1] < l_sort[-1] or l_shuf[-1] < l_shuf[0]
    
def test_batch_sizes():
    X, y = generate_labeled_data(100, 10, 2)
    res = compare_batch_sizes(X, y, epochs=5, batch_sizes=[8, 32], lr=0.01)
    assert 8 in res and 32 in res
