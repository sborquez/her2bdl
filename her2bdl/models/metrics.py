"""
Classification Metrics
======================

Metrics for multiclass classification.

[1] HAGHIGHI, Sepand, et al. PyCM: Multiclass confusion matrix library in Python. Journal of Open Source Software, 2018, vol. 3, no 25, p. 729.
"""

__all__ = [
    'confusion_matrix',
    'class_stat', 'overall_stat', 'multiclass_roc_curve'
]

import pycm
from sklearn.metrics import roc_curve, auc
import pandas as pd
from tensorflow.keras.utils import to_categorical

def confusion_matrix(y_true, y_pred, labels=None):
    cm = pycm.ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    if labels:
        if isinstance(labels, list):
            labels = {i:str(l) for i, l in enumerate(labels)}
        cm.relabel(mapping=labels)
    return cm


DEFAULT_CLASS_METRICS = [
    "TP", "FP", "TN", "FN",
    "PPV", # Precision
    "TPR", # Recall
    "ACC", # Accuracy
    "F1"   # F1 score
]

DEFAULT_OVERALL_METRICS = [
    "PPV Micro", # Precision micro
    "TPR Micro", # Recall micro
    "PPV Macro", # Precision macro
    "TPR Macro", # Recall macro
    "ACC Macro", # Average Accuracy 
    "ERR Macro", # Error rate
]

def class_stat(y_true, y_pred, labels=None, metrics=DEFAULT_CLASS_METRICS):
    cm = pycm.ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    if labels is not None:
        if isinstance(labels, list):
            labels = {i:str(l) for i, l in enumerate(labels)}
        cm.relabel(mapping=labels)
    stats = pd.DataFrame(cm.class_stat)[metrics]
    stats = stats.T.reset_index().rename(columns={"index": "class stat"})
    return stats

def overall_stat(y_true, y_pred, labels=None, metrics=DEFAULT_OVERALL_METRICS):
    cm = pycm.ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    if labels is not None:
        if isinstance(labels, list):
            labels = {i:str(l) for i, l in enumerate(labels)}
        cm.relabel(mapping=labels)
    stats = cm.overall_stat
    #add error rate
    stats["ERR Macro"] = sum(cm.ERR.values())/len(cm.ERR)
    stats = pd.DataFrame(stats)[metrics]
    stats = stats.T.drop(columns=[1]).reset_index().rename(columns={"index": "overall stat", 0: "value"})
    return stats

def multiclass_roc_curve(y_true, y_prob):
    m_samples, n_classes = y_prob.shape
    y_true = to_categorical(y_true, num_classes=n_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

