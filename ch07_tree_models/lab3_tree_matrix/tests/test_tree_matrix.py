import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from tree_matrix import *
from sklearn.tree import DecisionTreeClassifier

def test_tree_matrix():
    np.random.seed(42)
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X, y)
    
    A = build_matrix_A(clf, 3)
    B = build_matrix_B(clf)
    C = build_matrix_C(clf)
    D = build_matrix_D(clf)
    E = build_matrix_E(clf, 2)
    
    assert A.shape == (3, clf.tree_.node_count)
    assert B.shape == (1, clf.tree_.node_count)
    assert C.shape == (clf.tree_.node_count, clf.tree_.node_count)
    assert D.shape == (clf.tree_.node_count,)
    assert E.shape == (clf.tree_.node_count, 2)
    
    preds = matrix_predict(X, A, B, C, D, E, tree=clf)
    assert preds.shape == (10,)
    assert compare_predictions(clf, X, A, B, C, D, E)
