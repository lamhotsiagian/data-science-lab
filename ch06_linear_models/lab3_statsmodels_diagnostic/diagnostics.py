import numpy as np
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

def check_normality(residuals):
    stat, p = stats.shapiro(residuals)
    return p

def check_heteroscedasticity(result):
    import statsmodels.stats.api as sms
    stat, p, f, fp = sms.het_breuschpagan(result.resid, result.model.exog)
    return p

def check_autocorrelation(result):
    from statsmodels.stats.stattools import durbin_watson
    return durbin_watson(result.resid)

def full_diagnostic_report(X, y):
    result = fit_ols_model(X, y)
    resid = result.resid
    return {
        "summary": extract_summary_components(result),
        "normality_p": check_normality(resid),
        "heteroscedasticity_p": check_heteroscedasticity(result),
        "autocorrelation_dw": check_autocorrelation(result)
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
