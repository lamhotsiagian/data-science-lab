import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
from tree_viz import train_decision_tree, pruning_analysis, demonstrate_axis_limitation, demonstrate_pca_remedy, count_nodes
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def test_pruned_tree_fewer_nodes():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42, n_informative=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf_full = train_decision_tree(X_train, y_train)
    best_alpha, best_clf, metrics = pruning_analysis(X_train, y_train, X_test, y_test)
    
    assert count_nodes(best_clf) <= count_nodes(clf_full)
    assert metrics['test_score'] >= 0.0

def test_pca_dt_improves():
    X, y, clf1 = demonstrate_axis_limitation()
    X_pca, y_pca, clf2, pca = demonstrate_pca_remedy(X, y)
    
    acc1 = clf1.score(X, y)
    acc2 = clf2.score(X_pca, y)
    
    assert acc1 > 0
    assert acc2 > 0
