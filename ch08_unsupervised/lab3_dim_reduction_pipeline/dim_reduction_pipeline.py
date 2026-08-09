import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

class DimReductionPipeline:
    def __init__(self):
        self.reducer = None
        self.clusterer = None
        
    def fit(self, X, y=None):
        pass
        
    def select_method(self, X):
        if X.shape[1] > 50:
            return 'pca'
        else:
            return 'tsne'
            
    def reduce(self, X, n_components):
        method = self.select_method(X)
        if method == 'pca':
            self.reducer = PCA(n_components=n_components, random_state=42)
        else:
            self.reducer = TSNE(n_components=n_components, random_state=42)
            
        return self.reducer.fit_transform(X)
        
    def cluster(self, X_reduced, method='kmeans', **kwargs):
        if method == 'kmeans':
            self.clusterer = KMeans(random_state=42, n_init=10, **kwargs)
        
        return self.clusterer.fit_predict(X_reduced)
        
    def evaluate(self, X_reduced, labels):
        if len(np.unique(labels)) < 2:
            return {'silhouette': 0, 'calinski_harabasz': 0}
            
        sil = silhouette_score(X_reduced, labels)
        ch = calinski_harabasz_score(X_reduced, labels)
        return {'silhouette': sil, 'calinski_harabasz': ch}
        
    def fit_transform_cluster(self, X, n_components, cluster_method, **kwargs):
        X_reduced = self.reduce(X, n_components)
        labels = self.cluster(X_reduced, method=cluster_method, **kwargs)
        metrics = self.evaluate(X_reduced, labels)
        return X_reduced, labels, metrics

def compare_methods(X, y, methods, n_components):
    results = {}
    for name, method in methods.items():
        X_reduced = method.fit_transform(X)
        kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_reduced)
        sil = silhouette_score(X_reduced, labels) if len(np.unique(labels)) > 1 else 0
        results[name] = sil
    return results
