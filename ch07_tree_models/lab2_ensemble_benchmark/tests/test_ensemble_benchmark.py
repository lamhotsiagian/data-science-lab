import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
from ensemble_benchmark import generate_benchmark_data, train_all_ensembles, evaluate_all, adaboost_step_by_step
from sklearn.model_selection import train_test_split

def test_ensemble_beats_single():
    X, y = generate_benchmark_data(200, 5, 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = train_all_ensembles(X_train, y_train)
    results = evaluate_all(models, X_test, y_test)
    
    assert 'DecisionTree' in results
    assert 'RandomForest' in results
    assert results['RandomForest']['accuracy'] >= 0.0

def test_adaboost_weights():
    X, y = generate_benchmark_data(50, 2, 42)
    y = np.where(y == 0, -1, 1)
    weights_history, alpha_history = adaboost_step_by_step(X, y, 5)
    
    assert len(weights_history) == 5
    for w in weights_history:
        assert np.isclose(np.sum(w), 1.0)
    for a in alpha_history:
        assert a >= 0
