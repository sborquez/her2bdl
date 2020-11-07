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
def display_confusion_matrix(y_true, y_pred, 
                          model_name=None,
                          target_names=None, 
                          cm = None,
                          cmap=None,
                          normalize=True,
                          show=False,
                          save_to=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    y_true        targets.

    y_pred:       model's predictions.

    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
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
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = cm or confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    if model_name is not None:
        title = f"{model_name} - Confusion Matrix\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}"
    title = f"Confusion Matrix\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}"
    plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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
