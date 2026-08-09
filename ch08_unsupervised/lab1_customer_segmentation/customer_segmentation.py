import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd

def generate_customer_data(n_customers=500):
    X, _ = make_blobs(n_samples=n_customers, centers=4, n_features=5, cluster_std=1.0, random_state=42)
    df = pd.DataFrame(X, columns=['annual_income', 'spending_score', 'age', 'purchase_frequency', 'avg_basket_size'])
    return df

def run_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

def run_dbscan(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels

def run_hierarchical(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return model, labels

def find_optimal_k(X, k_range):
    inertias = []
    silhouettes = []
    
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(0)
            
    return inertias, silhouettes

def profile_segments(df, labels):
    df_copy = df.copy()
    df_copy['segment'] = labels
    return df_copy.groupby('segment').mean()

def plot_elbow(inertias, k_range):
    fig, ax = plt.subplots()
    ax.plot(k_range, inertias, marker='o')
    return fig, ax

def plot_silhouette(X, labels):
    pass

def plot_clusters_2d(X, labels, method_name):
    fig, ax = plt.subplots()
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
    ax.set_title(method_name)
    return fig, ax

def plot_segment_profiles(profiles):
    pass
