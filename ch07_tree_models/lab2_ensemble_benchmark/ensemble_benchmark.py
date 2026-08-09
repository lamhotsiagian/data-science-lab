import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

def generate_benchmark_data(n_samples=1000, n_features=20, random_state=42):
    n_informative = min(n_features, 2)
    return make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, n_redundant=0,
                               n_repeated=0, random_state=random_state)

def train_all_ensembles(X_train, y_train):
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(algorithm="SAMME", random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    except ImportError:
        pass
        
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    return models

def evaluate_all(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        start = time.time()
        preds = model.predict(X_test)
        end = time.time()
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        results[name] = {'accuracy': acc, 'f1': f1, 'time': end - start}
    return results

def adaboost_step_by_step(X, y, n_estimators):
    """Trace AdaBoost.M1 by hand so the weight dynamics are inspectable.

    The exponential update requires labels in {-1, +1}. `make_classification`
    and most sklearn datasets emit {0, 1}, so the labels are remapped before
    the update; using {0, 1} directly makes exp(-alpha*y*h) collapse to 1 for
    every sample with y = 0 and silently breaks reweighting.
    """
    n_samples = X.shape[0]
    weights = np.ones(n_samples) / n_samples
    weights_history = []
    alpha_history = []

    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("adaboost_step_by_step supports binary targets only")
    # Map the two observed classes onto {-1, +1} for the exponential update.
    y_signed = np.where(y == classes[0], -1.0, 1.0)

    for _ in range(n_estimators):
        weights_history.append(weights.copy())
        clf = DecisionTreeClassifier(max_depth=1, random_state=42)
        clf.fit(X, y, sample_weight=weights)
        preds = clf.predict(X)
        preds_signed = np.where(preds == classes[0], -1.0, 1.0)

        incorrect = (preds != y)
        err = np.sum(weights[incorrect]) / np.sum(weights)

        # A perfect stump carries no further information; clamp instead of
        # dividing by zero, and stop reweighting.
        if err <= 0:
            alpha_history.append(1.0)
            break
        if err >= 0.5:
            # Worse than chance on the weighted sample: the ensemble is done.
            alpha_history.append(0.0)
            break

        alpha = 0.5 * np.log((1 - err) / err)
        alpha_history.append(alpha)

        weights *= np.exp(-alpha * y_signed * preds_signed)
        weights /= np.sum(weights)

    return weights_history, alpha_history

def plot_benchmark_results(results):
    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    fig, ax = plt.subplots()
    ax.bar(names, accs)
    ax.set_ylabel('Accuracy')
    return fig, ax

def plot_adaboost_weights(weights_history):
    fig, ax = plt.subplots()
    ax.plot(np.array(weights_history)[:, :5]) # Plot first 5 samples
    return fig, ax

def plot_learning_curves(models, X, y):
    pass
