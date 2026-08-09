import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

def train_decision_tree(X, y, max_depth=None):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf

def get_ccp_alphas(tree, X, y):
    path = tree.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X, y)
        clfs.append(clf)
    
    train_scores = [clf.score(X, y) for clf in clfs]
    return ccp_alphas, train_scores, clfs

def plot_ccp_alpha_accuracy(alphas, train_scores, test_scores):
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    return fig, ax

def plot_tree_structure(tree, feature_names=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_tree(tree, feature_names=feature_names, filled=True, ax=ax)
    return fig, ax

def count_nodes(tree):
    return tree.tree_.node_count

def demonstrate_axis_limitation():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, (200, 2))
    y = (X[:, 0] > X[:, 1]).astype(int)
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    return X, y, clf

def demonstrate_pca_remedy(X, y):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_pca, y)
    
    return X_pca, y, clf, pca

def pruning_analysis(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    
    best_idx = np.argmax(test_scores)
    best_alpha = ccp_alphas[best_idx]
    best_clf = clfs[best_idx]
    
    metrics = {
        'best_alpha': best_alpha,
        'train_score': train_scores[best_idx],
        'test_score': test_scores[best_idx],
        'nodes': count_nodes(best_clf)
    }
    return best_alpha, best_clf, metrics
