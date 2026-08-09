import numpy as np
from sklearn.linear_model import LinearRegression

def ols_fit(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return beta

def ols_predict(X, beta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(beta)

def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def compute_residuals(y_true, y_pred):
    return y_true - y_pred

class OLSRegression:
    def fit(self, X, y):
        self.beta = ols_fit(X, y)
        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]
        return self
    def predict(self, X):
        return ols_predict(X, self.beta)
    def score(self, X, y):
        return compute_r_squared(y, self.predict(X))

def compare_with_sklearn(X, y):
    model = OLSRegression()
    model.fit(X, y)
    sk_model = LinearRegression().fit(X, y)
    return {"custom_coef": model.coef_, "sklearn_coef": sk_model.coef_}
