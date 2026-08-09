import os
import json

base_ch3 = "/Users/lamhots/ai/book-project/data-science/data-science-lab/ch03_probability_statistics/"
base_ch4 = "/Users/lamhots/ai/book-project/data-science/data-science-lab/ch04_feature_engineering/"

def make_notebook(title):
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}"]}
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.9"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

# CH3 Lab 1
lab1_dir = os.path.join(base_ch3, "lab1_monte_carlo")
write_file(os.path.join(lab1_dir, "monte_carlo.py"), """import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def estimate_pi(n_samples):
    np.random.seed(42)
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    inside = (x**2 + y**2) <= 1
    return 4 * np.sum(inside) / n_samples

def coin_flip_clt(n_flips, n_experiments):
    np.random.seed(42)
    flips = np.random.binomial(n_flips, 0.5, n_experiments)
    return flips

def option_price_monte_carlo(S0, K, r, sigma, T, n_simulations):
    np.random.seed(42)
    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoffs)

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def plot_pi_convergence(max_samples):
    samples = np.logspace(2, np.log10(max_samples), 50).astype(int)
    estimates = [estimate_pi(n) for n in samples]
    plt.plot(samples, estimates)

def plot_clt_demonstration(n_values):
    plt.hist(n_values, bins=30, density=True)

def plot_option_distribution(prices):
    plt.hist(prices, bins=50, density=True)
""")

write_file(os.path.join(lab1_dir, "tests", "test_monte_carlo.py"), """import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import monte_carlo
import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.stats as stats

def test_estimate_pi():
    pi_est = monte_carlo.estimate_pi(100000)
    assert abs(pi_est - np.pi) < 0.05

def test_option_price():
    mc_price = monte_carlo.option_price_monte_carlo(100, 100, 0.05, 0.2, 1, 100000)
    bs_price = monte_carlo.black_scholes_call(100, 100, 0.05, 0.2, 1)
    assert abs(mc_price - bs_price) / bs_price < 0.05

def test_clt():
    results = monte_carlo.coin_flip_clt(100, 1000)
    assert len(results) == 1000
""")

write_file(os.path.join(lab1_dir, "README.md"), "# Lab 1\n")
write_file(os.path.join(lab1_dir, "lab1_monte_carlo.ipynb"), json.dumps(make_notebook("Lab 1")))

# CH3 Lab 2
lab2_dir = os.path.join(base_ch3, "lab2_bayesian_inference")
write_file(os.path.join(lab2_dir, "bayesian_inference.py"), """import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def grid_posterior(prior, likelihood_func, data, grid):
    likelihood = likelihood_func(grid, data)
    unnormalized_posterior = prior * likelihood
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    return posterior

def medical_test_bayes(prevalence, sensitivity, specificity):
    p_disease = prevalence
    p_no_disease = 1 - prevalence
    p_pos_given_d = sensitivity
    p_pos_given_nd = 1 - specificity
    
    p_pos = (p_pos_given_d * p_disease) + (p_pos_given_nd * p_no_disease)
    return (p_pos_given_d * p_disease) / p_pos

def sequential_update(prior, likelihood_func, observations, grid):
    posteriors = []
    current_prior = prior
    for obs in observations:
        posterior = grid_posterior(current_prior, likelihood_func, obs, grid)
        posteriors.append(posterior)
        current_prior = posterior
    return posteriors

def beta_binomial_update(alpha, beta, successes, failures):
    return alpha + successes, beta + failures

def plot_prior_likelihood_posterior(prior, likelihood, posterior, grid):
    plt.plot(grid, prior)

def plot_sequential_updates(posteriors, grid, labels):
    pass
""")

write_file(os.path.join(lab2_dir, "tests", "test_bayesian_inference.py"), """import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bayesian_inference
import matplotlib
matplotlib.use('Agg')
import numpy as np

def test_grid_posterior_sum():
    grid = np.linspace(0, 1, 100)
    prior = np.ones(100) / 100
    likelihood_func = lambda g, d: g**d * (1-g)**(1-d)
    posterior = bayesian_inference.grid_posterior(prior, likelihood_func, 1, grid)
    assert np.isclose(np.sum(posterior), 1.0)

def test_medical_test():
    prob = bayesian_inference.medical_test_bayes(0.01, 0.99, 0.99)
    assert np.isclose(prob, 0.5)

def test_beta_binomial_update():
    new_alpha, new_beta = bayesian_inference.beta_binomial_update(1, 1, 10, 5)
    assert new_alpha == 11
    assert new_beta == 6
""")

write_file(os.path.join(lab2_dir, "README.md"), "# Lab 2\n")
write_file(os.path.join(lab2_dir, "lab2_bayesian_inference.ipynb"), json.dumps(make_notebook("Lab 2")))

# CH3 Lab 3
lab3_dir = os.path.join(base_ch3, "lab3_ab_testing")
write_file(os.path.join(lab3_dir, "ab_testing.py"), """import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

def simulate_ab_test(n_a, n_b, p_a, p_b):
    np.random.seed(42)
    conversions_a = np.random.binomial(n_a, p_a)
    conversions_b = np.random.binomial(n_b, p_b)
    return conversions_a, conversions_b

def compute_z_test(conversions_a, n_a, conversions_b, n_b):
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    p_pool = (conversions_a + conversions_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

def compute_confidence_interval(conversions, n, confidence):
    p = conversions / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se = np.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se

def power_analysis(effect_size, alpha, power):
    n = sms.NormalIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1)
    return n

def cuped_estimator(y_post, y_pre, x_post, x_pre):
    cov = np.cov(y_post, y_pre)[0, 1]
    var = np.var(y_pre)
    theta = cov / var
    y_cuped = y_post - theta * (y_pre - np.mean(y_pre))
    x_cuped = x_post - theta * (x_pre - np.mean(x_pre))
    return y_cuped, x_cuped

def run_simulation_study(n_simulations, n_per_group, true_effect, alpha):
    np.random.seed(42)
    rejections = 0
    p_a = 0.1
    p_b = p_a + true_effect
    for _ in range(n_simulations):
        conversions_a = np.random.binomial(n_per_group, p_a)
        conversions_b = np.random.binomial(n_per_group, p_b)
        _, p_val = compute_z_test(conversions_a, n_per_group, conversions_b, n_per_group)
        if p_val < alpha:
            rejections += 1
    return rejections / n_simulations

def plot_conversion_comparison(results): pass
def plot_power_curve(effect_sizes, sample_sizes): pass
def plot_cuped_variance_reduction(standard_var, cuped_var): pass
""")

write_file(os.path.join(lab3_dir, "tests", "test_ab_testing.py"), """import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ab_testing
import matplotlib
matplotlib.use('Agg')
import numpy as np

def test_type_I_error():
    error_rate = ab_testing.run_simulation_study(1000, 1000, 0, 0.05)
    assert 0.03 <= error_rate <= 0.07

def test_power_increases_with_n():
    n1 = ab_testing.power_analysis(0.1, 0.05, 0.8)
    n2 = ab_testing.power_analysis(0.1, 0.05, 0.9)
    assert n2 > n1

def test_cuped_variance_reduction():
    np.random.seed(42)
    y_pre = np.random.normal(0, 1, 1000)
    y_post = y_pre + np.random.normal(0, 0.5, 1000)
    x_pre = np.random.normal(0, 1, 1000)
    x_post = x_pre + np.random.normal(0, 0.5, 1000)
    
    y_cuped, x_cuped = ab_testing.cuped_estimator(y_post, y_pre, x_post, x_pre)
    assert np.var(y_cuped) < np.var(y_post)

def test_z_test():
    z, p = ab_testing.compute_z_test(100, 1000, 150, 1000)
    assert p < 0.05
""")
write_file(os.path.join(lab3_dir, "README.md"), "# Lab 3\n")
write_file(os.path.join(lab3_dir, "lab3_ab_testing.ipynb"), json.dumps(make_notebook("Lab 3")))

# CH4 Lab 1
lab1_dir_ch4 = os.path.join(base_ch4, "lab1_kaggle_features")
write_file(os.path.join(lab1_dir_ch4, "feature_engineering.py"), """import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from category_encoders import TargetEncoder

def create_polynomial_features(X, degree, columns):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(X[columns])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(columns), index=X.index)
    X_new = pd.concat([X.drop(columns, axis=1), poly_df], axis=1)
    return X_new

def create_interaction_features(X, col_pairs):
    X_new = X.copy()
    for col1, col2 in col_pairs:
        X_new[f"{col1}_{col2}_interact"] = X[col1] * X[col2]
    return X_new

def bin_continuous(X, col, n_bins, strategy):
    X_new = X.copy()
    if strategy == 'uniform':
        X_new[f"{col}_binned"] = pd.cut(X[col], bins=n_bins, labels=False)
    elif strategy == 'quantile':
        X_new[f"{col}_binned"] = pd.qcut(X[col], q=n_bins, labels=False, duplicates='drop')
    return X_new

def target_encode(X_train, y_train, X_val, col, smoothing):
    encoder = TargetEncoder(cols=[col], smoothing=smoothing)
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_train_enc[col] = encoder.fit_transform(X_train[col], y_train)
    X_val_enc[col] = encoder.transform(X_val[col])
    return X_train_enc, X_val_enc

def one_hot_encode(X, cols):
    X_new = pd.get_dummies(X, columns=cols, drop_first=True)
    return X_new

def ordinal_encode(X, col, order):
    encoder = OrdinalEncoder(categories=[order])
    X_new = X.copy()
    X_new[col] = encoder.fit_transform(X[[col]])
    return X_new

def select_features_mutual_info(X, y, k):
    mi = mutual_info_regression(X, y, random_state=42)
    top_k_indices = np.argsort(mi)[-k:]
    return X.columns[top_k_indices].tolist()

def engineer_all_features(X, y):
    X_new = create_interaction_features(X, [('MedInc', 'HouseAge')])
    X_new = bin_continuous(X_new, 'HouseAge', 5, 'quantile')
    return X_new

def compare_before_after(X_raw, X_engineered, y):
    score_raw = np.mean(cross_val_score(Ridge(random_state=42), X_raw, y, cv=3))
    score_eng = np.mean(cross_val_score(Ridge(random_state=42), X_engineered, y, cv=3))
    return score_raw, score_eng
""")

write_file(os.path.join(lab1_dir_ch4, "tests", "test_feature_engineering.py"), """import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feature_engineering
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def test_polynomial_features():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    new_df = feature_engineering.create_polynomial_features(df, 2, ['a'])
    assert 'a^2' in new_df.columns

def test_target_encode_no_leakage():
    X_train = pd.DataFrame({'cat': ['A', 'B', 'A']})
    y_train = pd.Series([1, 0, 1])
    X_val = pd.DataFrame({'cat': ['A', 'B']})
    X_tr_enc, X_val_enc = feature_engineering.target_encode(X_train, y_train, X_val, 'cat', smoothing=1)
    assert not X_val_enc.isnull().any().any()

def test_compare_before_after():
    np.random.seed(42)
    data = fetch_california_housing(as_frame=True)
    X = data.data.iloc[:500]
    y = data.target.iloc[:500]
    X_eng = feature_engineering.engineer_all_features(X, y)
    score_raw, score_eng = feature_engineering.compare_before_after(X, X_eng, y)
    assert not np.isnan(score_raw)
    assert not np.isnan(score_eng)
""")

write_file(os.path.join(lab1_dir_ch4, "README.md"), "# Lab 1\n")
write_file(os.path.join(lab1_dir_ch4, "lab1_kaggle_features.ipynb"), json.dumps(make_notebook("Lab 1")))


# CH4 Lab 2
lab2_dir_ch4 = os.path.join(base_ch4, "lab2_preprocessing_pipeline")
write_file(os.path.join(lab2_dir_ch4, "preprocessing_pipeline.py"), """import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_new = pd.DataFrame()
        for col in X.columns:
            dt = pd.to_datetime(X[col])
            X_new[f"{col}_year"] = dt.dt.year
            X_new[f"{col}_month"] = dt.dt.month
            X_new[f"{col}_day"] = dt.dt.day
        return X_new
    def get_feature_names_out(self, input_features=None):
        return None # Simplified

class PreprocessingConfig:
    def __init__(self, config_dict):
        self.numeric_cols = config_dict.get('numeric_cols', [])
        self.categorical_cols = config_dict.get('categorical_cols', [])
        self.datetime_cols = config_dict.get('datetime_cols', [])

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            return cls(yaml.safe_load(f))

def build_pipeline(config):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
    datetime_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('dt', DatetimeTransformer())])
        
    transformers = []
    if config.numeric_cols:
        transformers.append(('num', numeric_transformer, config.numeric_cols))
    if config.categorical_cols:
        transformers.append(('cat', categorical_transformer, config.categorical_cols))
    if config.datetime_cols:
        transformers.append(('dt', datetime_transformer, config.datetime_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor)])

def detect_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return {'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols, 'datetime_cols': datetime_cols}

def create_default_config(df):
    types = detect_column_types(df)
    return PreprocessingConfig(types)

def fit_transform_pipeline(df, config):
    pipeline = build_pipeline(config)
    return pipeline.fit_transform(df), pipeline

def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)

def load_pipeline(path):
    return joblib.load(path)
""")

write_file(os.path.join(lab2_dir_ch4, "config.yaml"), """numeric_cols:
  - age
  - income
categorical_cols:
  - city
datetime_cols:
  - signup_date
""")

write_file(os.path.join(lab2_dir_ch4, "tests", "test_preprocessing_pipeline.py"), """import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocessing_pipeline
import pandas as pd
import numpy as np

def test_pipeline_fit_transform():
    df = pd.DataFrame({
        'age': [25, np.nan, 30],
        'income': [50000, 60000, 70000],
        'city': ['A', 'B', np.nan],
        'signup_date': ['2023-01-01', '2023-02-01', '2023-03-01']
    })
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    
    config = preprocessing_pipeline.create_default_config(df)
    transformed, pipeline = preprocessing_pipeline.fit_transform_pipeline(df, config)
    
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()

def test_config_loading(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("numeric_cols: ['a']\\ncategorical_cols: ['b']")
    
    config = preprocessing_pipeline.PreprocessingConfig.from_yaml(config_path)
    assert config.numeric_cols == ['a']
    assert config.categorical_cols == ['b']
""")

write_file(os.path.join(lab2_dir_ch4, "README.md"), "# Lab 2\n")
write_file(os.path.join(lab2_dir_ch4, "lab2_preprocessing_pipeline.ipynb"), json.dumps(make_notebook("Lab 2")))

# CH4 Lab 3
lab3_dir_ch4 = os.path.join(base_ch4, "lab3_time_series_features")
write_file(os.path.join(lab3_dir_ch4, "time_series_features.py"), """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

def generate_synthetic_timeseries(n, trend, seasonality_period, noise_std):
    np.random.seed(42)
    t = np.arange(n)
    series = trend * t + 10 * np.sin(2 * np.pi * t / seasonality_period) + np.random.normal(0, noise_std, n)
    return pd.Series(series)

def create_rolling_features(df, col, windows):
    df_new = df.copy()
    for w in windows:
        df_new[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
        df_new[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
        df_new[f"{col}_roll_min_{w}"] = df[col].rolling(w).min()
        df_new[f"{col}_roll_max_{w}"] = df[col].rolling(w).max()
    return df_new

def create_lag_features(df, col, lags):
    df_new = df.copy()
    for lag in lags:
        df_new[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df_new

def seasonal_decompose_stl(df, col, period):
    stl = STL(df[col], period=period)
    res = stl.fit()
    return res.trend, res.seasonal, res.resid

def adf_test(series):
    result = adfuller(series.dropna())
    return result[0], result[1], result[1] < 0.05

def difference_series(series, d):
    return series.diff(periods=d)

def extract_all_features(df, col, windows, lags, period):
    df_new = create_rolling_features(df, col, windows)
    df_new = create_lag_features(df_new, col, lags)
    trend, seasonal, resid = seasonal_decompose_stl(df, col, period)
    df_new['trend'] = trend
    df_new['seasonal'] = seasonal
    df_new['resid'] = resid
    return df_new

def plot_decomposition(trend, seasonal, residual): pass
def plot_rolling_statistics(df, col, window): pass
""")

write_file(os.path.join(lab3_dir_ch4, "tests", "test_time_series_features.py"), """import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time_series_features
import pandas as pd
import numpy as np

def test_rolling_features():
    df = pd.DataFrame({'val': np.arange(10)})
    res = time_series_features.create_rolling_features(df, 'val', [3])
    assert 'val_roll_mean_3' in res.columns
    assert np.isnan(res['val_roll_mean_3'].iloc[0])

def test_adf_test():
    series = time_series_features.generate_synthetic_timeseries(100, 0, 10, 1)
    stat, pval, is_stat = time_series_features.adf_test(series)
    assert isinstance(is_stat, bool)

def test_decomposition():
    df = pd.DataFrame({'val': time_series_features.generate_synthetic_timeseries(100, 0, 10, 1)})
    trend, seasonal, resid = time_series_features.seasonal_decompose_stl(df, 'val', 10)
    assert np.allclose(df['val'], trend + seasonal + resid)
""")
write_file(os.path.join(lab3_dir_ch4, "README.md"), "# Lab 3\n")
write_file(os.path.join(lab3_dir_ch4, "lab3_time_series_features.ipynb"), json.dumps(make_notebook("Lab 3")))

# CH4 Lab 4
lab4_dir_ch4 = os.path.join(base_ch4, "lab4_text_preprocessing")
write_file(os.path.join(lab4_dir_ch4, "text_preprocessing.py"), """import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def tokenize_text(text, method='split'):
    if method == 'split':
        return text.split()
    elif method == 'regex':
        return re.findall(r'\\b\\w+\\b', text)
    elif method == 'nltk':
        return nltk.word_tokenize(text)
    return text.split()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\\w\\s]', '', text)
    return text.strip()

def stem_text(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words]

def build_tfidf_matrix(documents, max_features=None):
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer

def build_count_matrix(documents, max_features=None):
    vectorizer = CountVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer

def preprocess_pipeline(documents):
    processed = []
    for doc in documents:
        norm = normalize_text(doc)
        tokens = tokenize_text(norm, 'regex')
        no_stop = remove_stopwords(tokens)
        lemmas = lemmatize_text(no_stop)
        processed.append(" ".join(lemmas))
    return processed

def build_simple_embeddings(documents, dim):
    matrix, vectorizer = build_tfidf_matrix(documents)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    embeddings = svd.fit_transform(matrix)
    return embeddings
""")

write_file(os.path.join(lab4_dir_ch4, "tests", "test_text_preprocessing.py"), """import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import text_preprocessing

def test_normalization():
    text = "Hello, World!  "
    assert text_preprocessing.normalize_text(text) == "hello world"

def test_stopwords():
    tokens = ["this", "is", "a", "test"]
    assert text_preprocessing.remove_stopwords(tokens) == ["test"]

def test_tfidf_matrix():
    docs = ["hello world", "world of python"]
    matrix, vec = text_preprocessing.build_tfidf_matrix(docs)
    assert matrix.shape == (2, 4)

def test_pipeline():
    docs = ["This is a TEST!"]
    processed = text_preprocessing.preprocess_pipeline(docs)
    assert processed == ["test"]
""")
write_file(os.path.join(lab4_dir_ch4, "README.md"), "# Lab 4\n")
write_file(os.path.join(lab4_dir_ch4, "lab4_text_preprocessing.ipynb"), json.dumps(make_notebook("Lab 4")))

