"""
Metrics Visualizations
======================

Generate plot for different metrics of models.

Here you can find training metrics, single model evaluation
and models comparations.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

__all__ = [
]

"""
Training Metrics
================
"""

def plot_model_training_history(history, training_time, model_name, epochs, save_to=None):
    """
    Generate plot for training and validation Loss from a models history.
    """
    fig = plt.figure(figsize=(12,6))
    epochs = [i for i in range(1, epochs+1)]
    plt.plot(epochs, history.history['loss'], "*--", label="Train")
    plt.plot(epochs, history.history['val_loss'], "*--", label="Validation")

    # Style
    plt.title(f'Model {model_name} Training Loss\n Training time {training_time} [min]')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xticks(epochs, rotation=-90)
    plt.grid()
    
    # Show or Save
    if save_to is not None:
        fig.savefig(join(save_to, f'{model_name} - Training Loss.png'))
        plt.close(fig)
    else:
        plt.show()

"""
Model Metrics
==================
"""
