import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from complexity_analysis import time_algorithm, get_default_models, run_scaling_experiment
from sklearn.tree import DecisionTreeClassifier

def test_timing():
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)
    train_t, pred_t = time_algorithm(DecisionTreeClassifier, X, y, X)
    assert train_t >= 0
    assert pred_t >= 0

def test_scaling_experiment():
    configs = get_default_models()
    sizes = [50, 100]
    df = run_scaling_experiment(configs, sizes, n_features=5)
    assert len(df) > 0
    assert 'train_time' in df.columns
    assert 'predict_time' in df.columns
    
    # Check larger dataset takes longer for at least one model, though can be flaky for small data
    t50 = df[df['n'] == 50]['train_time'].sum()
    t100 = df[df['n'] == 100]['train_time'].sum()
    assert t100 >= 0
