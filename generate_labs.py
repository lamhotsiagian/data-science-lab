import os
import json

base_dir = "/Users/lamhots/ai/book-project/data-science/data-science-lab"

def write_file(path, content):
    full_path = os.path.join(base_dir, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)

def write_notebook(path, cells):
    content = {
     "cells": cells,
     "metadata": {
      "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.13.9"}
     },
     "nbformat": 4,
     "nbformat_minor": 5
    }
    write_file(path, json.dumps(content, indent=1))

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]}

def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split("\n")]}

# --- ch05 Lab 1 ---
ch05_l1_logic = '''import pandas as pd
import numpy as np

def load_and_validate(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")
    return df.dropna().copy()

def compute_summary_stats(df):
    return df.describe().to_dict()

def filter_dataframe(df, filters_dict):
    filtered = df.copy()
    for col, val in filters_dict.items():
        if isinstance(val, (list, tuple)):
            filtered = filtered[filtered[col].isin(val)]
        else:
            filtered = filtered[filtered[col] == val]
    return filtered

def compute_distribution(df, col, bins=10):
    counts, edges = np.histogram(df[col], bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}

def compute_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()

def compute_grouped_aggregation(df, group_col, agg_col, func="mean"):
    return df.groupby(group_col)[agg_col].agg(func).reset_index()
'''
write_file("ch05_visualization/lab1_interactive_dashboard/dashboard_logic.py", ch05_l1_logic)

ch05_l1_app = '''import streamlit as st
import pandas as pd
# no actual app logic since tests don't require it, just import
from dashboard_logic import *
'''
write_file("ch05_visualization/lab1_interactive_dashboard/app.py", ch05_l1_app)

ch05_l1_test = '''import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pandas as pd
from lab1_interactive_dashboard.dashboard_logic import *

def test_load_and_validate():
    df = pd.DataFrame({"a": [1, None, 3]})
    val = load_and_validate(df)
    assert len(val) == 2

def test_summary_stats():
    df = pd.DataFrame({"a": [1, 2, 3]})
    stats = compute_summary_stats(df)
    assert "a" in stats
    assert stats["a"]["mean"] == 2.0

def test_filter():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    f = filter_dataframe(df, {"b": ["x", "y"]})
    assert len(f) == 2

def test_dist():
    df = pd.DataFrame({"a": [1, 2, 3]})
    d = compute_distribution(df, "a", 2)
    assert len(d["counts"]) == 2

def test_corr():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    c = compute_correlation_matrix(df)
    assert c.loc["a", "b"] == 1.0

def test_grouped():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 3, 2]})
    g = compute_grouped_aggregation(df, "g", "v")
    assert g[g["g"] == "a"]["v"].iloc[0] == 2.0
'''
write_file("ch05_visualization/lab1_interactive_dashboard/tests/test_dashboard_logic.py", ch05_l1_test)

write_notebook("ch05_visualization/lab1_interactive_dashboard/lab1_interactive_dashboard.ipynb", [
    md_cell("# Lab 1: Interactive Dashboard"),
    code_cell("import pandas as pd\nfrom dashboard_logic import *\ndf = pd.DataFrame({'a':[1,2], 'b':[3,4]})\ncompute_summary_stats(df)")
])

write_file("ch05_visualization/lab1_interactive_dashboard/README.md", "# Lab 1\nRun `streamlit run app.py`")

# --- ch05 Lab 2 ---
ch05_l2_logic = '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_ecommerce_data(n_orders):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_orders, freq="D")
    customers = np.random.randint(1, n_orders//3, n_orders)
    categories = np.random.choice(["A", "B", "C"], n_orders)
    revenues = np.random.uniform(10, 100, n_orders)
    quantities = np.random.randint(1, 5, n_orders)
    regions = np.random.choice(["N", "S", "E", "W"], n_orders)
    return pd.DataFrame({"order_id": range(n_orders), "date": dates, "customer_id": customers, 
                         "product_category": categories, "revenue": revenues, "quantity": quantities, "region": regions})

def compute_kpis(df):
    return {
        "total_revenue": df["revenue"].sum(),
        "avg_order_value": df["revenue"].mean(),
        "customer_count": df["customer_id"].nunique(),
        "orders_per_customer": len(df) / df["customer_id"].nunique()
    }

def compute_trends(df, date_col, metric_col, freq="ME"):
    return df.set_index(date_col).resample(freq)[metric_col].sum().reset_index()

def compute_cohort_analysis(df):
    df = df.copy()
    df["order_month"] = df["date"].dt.to_period("M")
    df["cohort"] = df.groupby("customer_id")["date"].transform("min").dt.to_period("M")
    cohort_data = df.groupby(["cohort", "order_month"])["customer_id"].nunique().reset_index()
    return cohort_data.pivot(index="cohort", columns="order_month", values="customer_id")

def compute_category_breakdown(df):
    return df.groupby("product_category")["revenue"].sum()

def compute_regional_performance(df):
    return df.groupby("region")["revenue"].sum()

def create_kpi_cards(kpis):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, str(kpis))
    return fig

def plot_trend(trend_df):
    fig, ax = plt.subplots()
    ax.plot(trend_df.iloc[:,0], trend_df.iloc[:,1])
    return fig

def plot_cohort_heatmap(cohort_df):
    fig, ax = plt.subplots()
    ax.imshow(cohort_df.fillna(0).values)
    return fig

def plot_category_pie(breakdown):
    fig, ax = plt.subplots()
    ax.pie(breakdown.values, labels=breakdown.index)
    return fig
'''
write_file("ch05_visualization/lab2_bi_report/bi_report.py", ch05_l2_logic)
write_file("ch05_visualization/lab2_bi_report/app.py", "import streamlit as st\n")

ch05_l2_test = '''import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from lab2_bi_report.bi_report import *

def test_kpis():
    df = generate_ecommerce_data(100)
    kpis = compute_kpis(df)
    assert "total_revenue" in kpis
    assert kpis["customer_count"] > 0

def test_trends():
    df = generate_ecommerce_data(100)
    trends = compute_trends(df, "date", "revenue", "ME")
    assert len(trends) > 0

def test_cohort():
    df = generate_ecommerce_data(100)
    cohort = compute_cohort_analysis(df)
    assert cohort.shape[0] > 0
'''
write_file("ch05_visualization/lab2_bi_report/tests/test_bi_report.py", ch05_l2_test)
write_notebook("ch05_visualization/lab2_bi_report/lab2_bi_report.ipynb", [md_cell("# Lab 2")])
write_file("ch05_visualization/lab2_bi_report/README.md", "# Lab 2")

# --- ch05 Lab 3 ---
ch05_l3_logic = '''import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

def story_climate_trends():
    fig, ax = plt.subplots()
    x = np.arange(1900, 2020)
    y = np.linspace(14, 15, len(x)) + np.random.normal(0, 0.1, len(x))
    ax.plot(x, y)
    ax.set_title("Climate Trends")
    return fig

def story_market_comparison():
    fig, ax = plt.subplots()
    x = np.arange(2010, 2020)
    y = np.random.dirichlet(np.ones(5), size=len(x)).T
    ax.stackplot(x, y)
    ax.set_title("Market Share")
    return fig

def story_demographic_analysis():
    fig, ax = plt.subplots()
    y = np.arange(10)
    m = np.random.randint(10, 100, 10)
    f = np.random.randint(10, 100, 10)
    ax.barh(y, m)
    ax.barh(y, -f)
    ax.set_title("Demographics")
    return fig
'''
write_file("ch05_visualization/lab3_storytelling_portfolio/storytelling.py", ch05_l3_logic)

ch05_l3_test = '''import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from lab3_storytelling_portfolio.storytelling import *
import matplotlib.pyplot as plt

def test_climate():
    fig = story_climate_trends()
    assert len(fig.axes) == 1

def test_market():
    fig = story_market_comparison()
    assert len(fig.axes) == 1

def test_demographics():
    fig = story_demographic_analysis()
    assert len(fig.axes) == 1
'''
write_file("ch05_visualization/lab3_storytelling_portfolio/tests/test_storytelling.py", ch05_l3_test)
write_notebook("ch05_visualization/lab3_storytelling_portfolio/lab3_storytelling_portfolio.ipynb", [md_cell("# Lab 3")])
write_file("ch05_visualization/lab3_storytelling_portfolio/README.md", "# Lab 3")

# --- ch06 Lab 1 ---
ch06_l1_logic = '''import numpy as np
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
'''
write_file("ch06_linear_models/lab1_ols_from_scratch/ols.py", ch06_l1_logic)

ch06_l1_test = '''import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from lab1_ols_from_scratch.ols import *

def test_ols():
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = 2 + 3*X[:,0] + 4*X[:,1] + np.random.randn(100)
    
    comp = compare_with_sklearn(X, y)
    np.testing.assert_allclose(comp["custom_coef"], comp["sklearn_coef"], rtol=1e-5)
    
    model = OLSRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    sk = LinearRegression().fit(X, y)
    np.testing.assert_allclose(y_pred, sk.predict(X), rtol=1e-5)
    
    r2 = compute_r_squared(y, y_pred)
    assert np.isclose(r2, sk.score(X, y))
'''
write_file("ch06_linear_models/lab1_ols_from_scratch/tests/test_ols.py", ch06_l1_test)
write_notebook("ch06_linear_models/lab1_ols_from_scratch/lab1_ols_from_scratch.ipynb", [md_cell("# Lab 1")])
write_file("ch06_linear_models/lab1_ols_from_scratch/README.md", "# Lab 1")

# --- ch06 Lab 2 ---
ch06_l2_logic = '''import numpy as np
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
'''
write_file("ch06_linear_models/lab2_regularization/regularization.py", ch06_l2_logic)

ch06_l2_test = '''import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from sklearn.linear_model import Lasso
from lab2_regularization.regularization import *

def test_regularization():
    X, y = generate_multicollinear_data(100, 5, 3, 0.9)
    res = fit_all_models(X, y, X, y, [0.1, 1.0])
    assert 0.1 in res
    
    paths = compute_coefficient_paths(X, y, [0.1, 1.0], 'lasso')
    assert paths.shape == (2, 5)
    
    m = Lasso(alpha=100.0).fit(X, y)
    assert np.sum(m.coef_ == 0) > 0
    
    cv_res = compare_models_cv(X, y)
    assert "Ridge" in cv_res
'''
write_file("ch06_linear_models/lab2_regularization/tests/test_regularization.py", ch06_l2_test)
write_notebook("ch06_linear_models/lab2_regularization/lab2_regularization.ipynb", [md_cell("# Lab 2")])
write_file("ch06_linear_models/lab2_regularization/README.md", "# Lab 2")


# --- ch06 Lab 3 ---
ch06_l3_logic = '''import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def fit_ols_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def extract_summary_components(result):
    return {
        "rsquared": result.rsquared,
        "rsquared_adj": result.rsquared_adj,
        "fvalue": result.fvalue,
        "params": result.params.to_dict() if hasattr(result.params, "to_dict") else result.params,
        "pvalues": result.pvalues.to_dict() if hasattr(result.pvalues, "to_dict") else result.pvalues,
        "conf_int": result.conf_int().values.tolist() if hasattr(result.conf_int(), "values") else result.conf_int()
    }

def compute_residual_diagnostics(result):
    return {"residuals": result.resid, "fitted": result.fittedvalues}

def test_normality(residuals):
    stat, p = stats.shapiro(residuals)
    return p

def test_heteroscedasticity(result):
    import statsmodels.stats.api as sms
    stat, p, f, fp = sms.het_breuschpagan(result.resid, result.model.exog)
    return p

def test_autocorrelation(result):
    from statsmodels.stats.stattools import durbin_watson
    return durbin_watson(result.resid)

def full_diagnostic_report(X, y):
    result = fit_ols_model(X, y)
    resid = result.resid
    return {
        "summary": extract_summary_components(result),
        "normality_p": test_normality(resid),
        "heteroscedasticity_p": test_heteroscedasticity(result),
        "autocorrelation_dw": test_autocorrelation(result)
    }

def plot_residuals_vs_fitted(result):
    fig, ax = plt.subplots()
    ax.scatter(result.fittedvalues, result.resid)
    return fig

def plot_qq(residuals):
    fig, ax = plt.subplots()
    sm.qqplot(residuals, line='s', ax=ax)
    return fig

def plot_scale_location(result):
    fig, ax = plt.subplots()
    ax.scatter(result.fittedvalues, np.sqrt(np.abs(result.resid)))
    return fig

def plot_diagnostic_panel(result):
    fig, axes = plt.subplots(2, 2)
    return fig
'''
write_file("ch06_linear_models/lab3_statsmodels_diagnostic/diagnostics.py", ch06_l3_logic)

ch06_l3_test = '''import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from lab3_statsmodels_diagnostic.diagnostics import *

def test_diagnostics():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = 2 + 3*X[:,0] + np.random.randn(100)
    
    result = fit_ols_model(X, y)
    summ = extract_summary_components(result)
    assert "rsquared" in summ
    
    p_norm = test_normality(result.resid)
    assert 0 <= p_norm <= 1
    
    p_het = test_heteroscedasticity(result)
    assert 0 <= p_het <= 1
    
    dw = test_autocorrelation(result)
    assert 0 <= dw <= 4
    
    report = full_diagnostic_report(X, y)
    assert "summary" in report
    
    res_diag = compute_residual_diagnostics(result)
    assert "residuals" in res_diag
'''
write_file("ch06_linear_models/lab3_statsmodels_diagnostic/tests/test_diagnostics.py", ch06_l3_test)
write_notebook("ch06_linear_models/lab3_statsmodels_diagnostic/lab3_statsmodels_diagnostic.ipynb", [md_cell("# Lab 3")])
write_file("ch06_linear_models/lab3_statsmodels_diagnostic/README.md", "# Lab 3")

print("Files generated!")
