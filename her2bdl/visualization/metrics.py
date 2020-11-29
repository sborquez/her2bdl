"""
Metrics Visualizations
======================

Generate plot for different metrics of models.

Here you can find training metrics, single model evaluation
and models comparations.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix


__all__ = [
    'display_model_training_history',
    'display_confusion_matrix'
]

"""
Training Metrics
================
"""

def display_model_training_history(history, training_time, model_name, epochs, show=False, save_to=None):
    """
    Generate display for training and validation Loss from a models history.
    """
    fig = plt.figure(figsize=(12,6))
    epochs = [i for i in range(1, epochs+1)]
    epochs_recorded = epochs[:len(history.history['loss'])]
    plt.plot(epochs_recorded, history.history['loss'], "*--", label="Train")
    plt.plot(epochs_recorded, history.history['val_loss'], "*--", label="Validation")

    # Style
    plt.title(f'Model {model_name} Training Loss\n Training time {training_time} [min]')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xticks(epochs_recorded, rotation=-90)
    plt.grid()
    
    # Show or Save
    if save_to is not None:
        save_to = Path(save_to)
        fig.savefig(save_to.joinpath(f'{model_name} - Training Loss.png'))
    if show:
        plt.show()
        plt.close()
        return None
    else:
        return fig

"""
Model Metrics
==================
"""
def display_confusion_matrix(y_true, y_pred, model_name=None, labels=None, 
                            cm=None, cmap=None, normalize=True, show=False,
                            save_to=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    y_true        targets.
    y_pred:       model's predictions.
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    labels: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Return
    ------
        figure or None

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          labels = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = cm or confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if normalize:
        group_counts = [f"{value}"
                        for value in cm.flatten()]
        group_percentages = [f"{100*value:.1f}"
                             for value in cm.flatten()/np.sum(cm)]
        annot = [f"{v1}\n{v2}%"
                 for v1, v2 in zip(group_counts, group_percentages)]
        annot = np.array(annot).reshape(cm.shape)
        fmt =''
    else:
        fmt ='d'
        annot=True
    # Heatmap
    fig = plt.figure(figsize=(5, 4))
    labels = labels or sorted(np.unique(y_true))
    cmap = cmap or plt.get_cmap('Blues')  
    sns.heatmap(cm, annot=annot, cmap=cmap, fmt=fmt, 
                xticklabels=labels, yticklabels=labels)
    # Style
    if model_name is not None:
        title = f"{model_name}\nConfusion Matrix"
    else:
        title = f"Confusion Matrix"
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\n\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}')
    plt.grid(False)
    plt.tight_layout()
    # Show or Save
    if save_to is not None:
        save_to = Path(save_to)
        fig.savefig(save_to.joinpath(f'{model_name}_Confusion_ Matrix.png'))
    if show:
        plt.show()
        plt.close()
        return None
    else:
        return fig