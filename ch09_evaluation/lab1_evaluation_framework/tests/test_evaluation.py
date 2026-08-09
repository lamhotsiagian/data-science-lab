import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from evaluation import ModelEvaluator

def test_classification_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2])
    
    res = ModelEvaluator.classification_metrics(y_true, y_pred, y_prob)
    assert res['accuracy'] == 0.75
    
def test_perfect_predictions():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_prob = np.array([0.0, 1.0, 1.0, 0.0])
    
    res = ModelEvaluator.classification_metrics(y_true, y_pred, y_prob)
    assert res['accuracy'] == 1.0
    assert res['auc_roc'] == 1.0

def test_random_predictions():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 1000)
    y_prob = rng.random(1000)
    y_pred = (y_prob > 0.5).astype(int)
    
    res = ModelEvaluator.classification_metrics(y_true, y_pred, y_prob)
    assert 0.4 < res['auc_roc'] < 0.6
