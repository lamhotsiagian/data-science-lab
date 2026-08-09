import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve

class ModelEvaluator:
    @staticmethod
    def classification_metrics(y_true, y_pred, y_prob):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_prob),
            'auc_pr': average_precision_score(y_true, y_prob)
        }
        
    @staticmethod
    def regression_metrics(y_true, y_pred):
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
        
    @staticmethod
    def confusion_matrix_analysis(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        return cm, per_class_accuracy
        
    @staticmethod
    def compute_roc_curve(y_true, y_prob):
        return roc_curve(y_true, y_prob)
        
    @staticmethod
    def compute_pr_curve(y_true, y_prob):
        return precision_recall_curve(y_true, y_prob)
        
    @staticmethod
    def compute_calibration(y_true, y_prob, n_bins=5):
        return calibration_curve(y_true, y_prob, n_bins=n_bins)

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label')
    return fig

def plot_roc_curve(fpr, tpr, auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.legend()
    return fig

def plot_pr_curve(precision, recall, auc):
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'AUC = {auc:.2f}')
    ax.legend()
    return fig

def plot_calibration_curve(y_true, y_prob):
    prob_true, prob_pred = ModelEvaluator.compute_calibration(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker='o')
    ax.plot([0, 1], [0, 1], linestyle='--')
    return fig

def plot_regression_diagnostics(y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_true, y_pred)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0].set_xlabel('True')
    axes[0].set_ylabel('Predicted')
    
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')
    return fig

def full_classification_report(y_true, y_pred, y_prob, classes):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    cm = confusion_matrix(y_true, y_pred)
    axes[0, 0].imshow(cm, cmap=plt.cm.Blues)
    axes[0, 0].set_title('Confusion Matrix')
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    axes[0, 1].plot(fpr, tpr, label=f'AUC={auc_roc:.2f}')
    axes[0, 1].plot([0, 1], [0, 1], 'r--')
    axes[0, 1].legend()
    axes[0, 1].set_title('ROC Curve')
    
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    axes[1, 0].plot(rec, prec, label=f'AUC={auc_pr:.2f}')
    axes[1, 0].legend()
    axes[1, 0].set_title('PR Curve')
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
    axes[1, 1].plot(prob_pred, prob_true, marker='o')
    axes[1, 1].plot([0, 1], [0, 1], 'r--')
    axes[1, 1].set_title('Calibration')
    
    plt.tight_layout()
    return fig
