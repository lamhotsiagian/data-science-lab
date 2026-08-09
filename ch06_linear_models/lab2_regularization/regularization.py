import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_multicollinear_data(n, n_features, n_informative, collinearity_factor):
    X, y = make_regression(n_samples=n, n_features=n_features, n_informative=n_informative, noise=1.0, random_state=42)
    if n_features > 1:
        X[:, -1] = X[:, 0] * collinearity_factor + np.random.normal(0, 0.1, n)
    return X, y

def fit_all_models(X_train, y_train, X_test, y_test, alphas):
    res = {}
    for a in alphas:
        r = Ridge(alpha=a).fit(X_train, y_train)
        l = Lasso(alpha=a).fit(X_train, y_train)
        e = ElasticNet(alpha=a).fit(X_train, y_train)
        res[a] = {"Ridge": r.score(X_test, y_test), "Lasso": l.score(X_test, y_test), "ElasticNet": e.score(X_test, y_test)}
    return res

def compute_coefficient_paths(X, y, alphas, model_type):
    coefs = []
    for a in alphas:
        if model_type == 'ridge':
            m = Ridge(alpha=a).fit(X, y)
        elif model_type == 'lasso':
            m = Lasso(alpha=a).fit(X, y)
        else:
            m = ElasticNet(alpha=a).fit(X, y)
        coefs.append(m.coef_)
    return np.array(coefs)

def compare_models_cv(X, y):
    rcv = RidgeCV(cv=3).fit(X, y)
    lcv = LassoCV(cv=3).fit(X, y)
    ecv = ElasticNetCV(cv=3).fit(X, y)
    return {"Ridge": rcv.score(X, y), "Lasso": lcv.score(X, y), "ElasticNet": ecv.score(X, y)}

def plot_coefficient_paths(paths, alphas):
    fig, ax = plt.subplots()
    ax.plot(alphas, paths)
    ax.set_xscale('log')
    return fig

def plot_model_comparison(results):
    fig, ax = plt.subplots()
    return fig

def plot_regularization_effect_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig
