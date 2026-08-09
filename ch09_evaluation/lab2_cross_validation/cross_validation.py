import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

def kfold_cv(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(X):
        m = deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
    return scores

def stratified_kfold_cv(model, X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        m = deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
    return scores

def loocv(model, X, y):
    loo = LeaveOneOut()
    scores = []
    for train_idx, test_idx in loo.split(X):
        m = deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
    return scores

def time_series_cv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        m = deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
    return scores
    
def blocked_time_series_cv(model, X, y, n_splits=5, gap=0):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        m = deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
    return scores

def compare_cv_methods(model, X, y):
    res = {
        'kfold': np.var(kfold_cv(model, X, y, 5)),
        'stratified': np.var(stratified_kfold_cv(model, X, y, 5))
    }
    return res

def demonstrate_data_leakage(X, y):
    from sklearn.linear_model import LogisticRegression
    # Leaked
    scaler = StandardScaler()
    X_scaled_leaked = scaler.fit_transform(X)
    model = LogisticRegression()
    leaked_scores = kfold_cv(model, X_scaled_leaked, y, k=5)
    
    # Correct
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    correct_scores = []
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        m = LogisticRegression()
        m.fit(X_train, y[train_idx])
        correct_scores.append(accuracy_score(y[test_idx], m.predict(X_test)))
        
    return leaked_scores, correct_scores

def plot_cv_comparison(results):
    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values())
    return fig

def plot_fold_scores(scores_per_fold, method_name):
    fig, ax = plt.subplots()
    ax.plot(scores_per_fold, marker='o')
    ax.set_title(method_name)
    return fig

def plot_leakage_comparison(leaked_scores, correct_scores):
    fig, ax = plt.subplots()
    ax.boxplot([leaked_scores, correct_scores], labels=['Leaked', 'Correct'])
    return fig
