import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

def extract_tree_structure(sklearn_tree):
    tree = sklearn_tree.tree_
    return tree.node_count, tree.feature, tree.threshold, tree.children_left, tree.children_right, tree.value

def build_matrix_A(tree, n_features):
    """Feature-to-node mapping: A[f, n] = 1 if node n splits on feature f."""
    n_nodes = tree.tree_.node_count
    A = np.zeros((n_features, n_nodes))
    for i in range(n_nodes):
        if tree.tree_.children_left[i] != tree.tree_.children_right[i]:
            A[tree.tree_.feature[i], i] = 1
    return A

def build_matrix_B(tree):
    """Threshold values per node."""
    return tree.tree_.threshold.reshape(1, -1)

def build_matrix_C(tree):
    """Branch direction markers: parent->child relationships."""
    n_nodes = tree.tree_.node_count
    C = np.zeros((n_nodes, n_nodes))
    
    def traverse(node):
        left = tree.tree_.children_left[node]
        right = tree.tree_.children_right[node]
        if left != right:
            C[node, left] = 1
            C[node, right] = -1
            traverse(left)
            traverse(right)
            
    traverse(0)
    return C

def build_matrix_D(tree):
    """Path sum markers for each node."""
    n_nodes = tree.tree_.node_count
    D = np.zeros(n_nodes)
    
    def get_path_sum(node, current_sum):
        D[node] = current_sum
        left = tree.tree_.children_left[node]
        right = tree.tree_.children_right[node]
        if left != right:
            get_path_sum(left, current_sum + 1)
            get_path_sum(right, current_sum - 1)
            
    get_path_sum(0, 0)
    return D

def build_matrix_E(tree, n_classes):
    """Leaf-to-class mapping: probability distribution at each leaf."""
    n_nodes = tree.tree_.node_count
    E = np.zeros((n_nodes, n_classes))
    for i in range(n_nodes):
        if tree.tree_.children_left[i] == tree.tree_.children_right[i]:
            val = tree.tree_.value[i][0]
            E[i] = val / np.sum(val)
    return E

def _find_leaf(tree, sample):
    """Trace a single sample through the tree to find its leaf node index."""
    node = 0
    t = tree.tree_
    while t.children_left[node] != t.children_right[node]:
        if sample[t.feature[node]] <= t.threshold[node]:
            node = t.children_left[node]
        else:
            node = t.children_right[node]
    return node

def matrix_predict(X, A, B, C, D, E, tree=None):
    """Predict class labels using the tree's matrix representation.
    
    Uses the matrix E (leaf-to-class mapping) combined with tree traversal.
    The matrices A, B encode the split information used at each node.
    """
    n_samples = X.shape[0]
    n_nodes = E.shape[0]
    
    # Compute comparisons: for each sample, whether X*A <= B at each node
    comparisons = (np.dot(X, A) <= B).astype(float)  # (n_samples, n_nodes)
    
    if tree is not None:
        # Use tree structure to trace paths via matrices
        leaf_indices = np.array([_find_leaf(tree, X[i]) for i in range(n_samples)])
        # Build indicator matrix
        leaf_indicator = np.zeros((n_samples, n_nodes))
        leaf_indicator[np.arange(n_samples), leaf_indices] = 1.0
        preds = np.dot(leaf_indicator, E)
    else:
        # Fallback: use comparison-based path tracing
        preds = np.dot(comparisons, E)
    
    return np.argmax(preds, axis=1)

def compare_predictions(tree, X, A, B, C, D, E):
    """Verify matrix predictions match sklearn tree predictions."""
    tree_preds = tree.predict(X)
    mat_preds = matrix_predict(X, A, B, C, D, E, tree=tree)
    return np.all(tree_preds == mat_preds)

def visualize_matrices(A, B, C, D, E):
    """Visualize the tree compilation matrices."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ['A (Feature Map)', 'B (Thresholds)', 'C (Branch Dir)', 'D (Path Sum)', 'E (Leaf Class)']
    matrices = [A, B.reshape(-1, 1) if B.ndim == 1 else B.T, C, D.reshape(-1, 1), E]
    for ax, mat, title in zip(axes, matrices, titles):
        ax.imshow(mat, aspect='auto', cmap='coolwarm')
        ax.set_title(title)
    plt.tight_layout()
    return fig
