import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification

def time_algorithm(model_class, X_train, y_train, X_test, **kwargs):
    model = model_class(**kwargs)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    t1 = time.time()
    model.predict(X_test)
    predict_time = time.time() - t1
    
    return train_time, predict_time

def get_default_models():
    return [
        ('LinearRegression', LinearRegression, {}),
        ('KNN', KNeighborsClassifier, {'n_neighbors': 5}),
        ('DecisionTree', DecisionTreeClassifier, {}),
        ('RandomForest', RandomForestClassifier, {'n_estimators': 10}),
        ('SVM', SVC, {'kernel': 'rbf'})
    ]

def run_scaling_experiment(model_configs, dataset_sizes, n_features=20, random_state=42):
    results = []
    for n in dataset_sizes:
        X, y = make_classification(n_samples=n*2, n_features=n_features, random_state=random_state)
        X_train, y_train = X[:n], y[:n]
        X_test = X[n:]
        for name, cls, kwargs in model_configs:
            if name == 'LinearRegression': continue
            tr_t, pr_t = time_algorithm(cls, X_train, y_train, X_test, **kwargs)
            results.append({
                'model': name,
                'n': n,
                'train_time': tr_t,
                'predict_time': pr_t
            })
    return pd.DataFrame(results)

def plot_training_scaling(results):
    fig, ax = plt.subplots()
    for model in results['model'].unique():
        sub = results[results['model'] == model]
        ax.plot(sub['n'], sub['train_time'], label=model, marker='o')
    ax.legend()
    return fig

def plot_prediction_scaling(results):
    fig, ax = plt.subplots()
    for model in results['model'].unique():
        sub = results[results['model'] == model]
        ax.plot(sub['n'], sub['predict_time'], label=model, marker='o')
    ax.legend()
    return fig

def plot_scaling_comparison(results):
    fig, ax = plt.subplots()
    for model in results['model'].unique():
        sub = results[results['model'] == model]
        ax.loglog(sub['n'], sub['train_time'], label=model, marker='o')
    ax.legend()
    return fig
