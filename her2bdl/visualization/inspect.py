"""
Inspect Visualizations
======================

Generate plot inspect the blackbox. 

Here you can find weights distributios, activations plots and model plot
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

__all__ = [
    'plot_predictive_distribution'
]

def plot_predictive_distribution(predictive_distribution, target_names=None, axis=None):
    """
    Plot predictive distribution.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,8))
        axis = plt.gca()
    if target_names is not None:
        x = np.arange(len(target_names))
        ticks = [target_names[i] for i in x]
    else:
        x = list(range(len(predictive_distribution)))
        ticks = x
    axis.bar(x, predictive_distribution)
    axis.set_title("Predictive Distribution")
    # Style
    axis.set_xticks(x)
    axis.set_xticklabels(ticks, rotation=45)
    axis.set_xlabel("y")
    axis.set_ylabel("P(y)")
    axis.set_ylim([0, 1])
    axis.legend()
    return axis

def plot_forward_pass_samples(prediction_samples, target_names=None, axis=None):
    """
    Plot samples distribution for each class.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,8))
        axis = plt.gca()
    if target_names is not None:
        labels = [target_names[i] for i in np.arange(len(target_names))]
    else:
        labels = list(range(len(prediction_samples.T)))
    df = pd.DataFrame(prediction_samples, columns=labels)
    sns.histplot(data=df, ax=axis, stat="density", kde=True, binrange=(0,1), legend=True)
    # Style
    axis.set_title("Samples Class Distributions")
    axis.set_xlabel("class probability")
    axis.set_ylabel("samples density")
    return axis

def plot_sample(x, y_pred, y_true):
    pass

def display_uncertainty(x, y_pred, predictive_distribution, prediction_samples, y_true=None, uncertainty={}):
    """
    For one input x, display predictive distribution and samples class distributions.
    """
    pass

def display_prediction(x, y_pred, model_name, y_true=None, show=False, save_to=None):
    """
    Display the prediction of a input, the probability and the predicted point.
    If targets_values is not None, the target point is included in the figure.
    """
    pass
    # # Create new Figure
    # fig = plt.figure(figsize=(8,8))
    # ax = plt.gca()
    
    # # Style
    # plt.title(title)

    # # point estimator
    # if np.array_equal(prediction, prediction_point):
    #     if len(targets) == 1:
    #         ax = show_points_1d(prediction, prediction_point, targets, target_domains, targets_values, ax)
    #     elif len(targets) == 2:
    #         ax = show_points_2d(prediction, prediction_point, targets, target_domains, targets_values, ax)
    #     elif len(targets) == 3:
    #         raise NotImplementedError

    # # Show or Save
    # if save_to is not None:
    #     save_to = Path(save_to)
    #     fig.savefig(save_to.joinpath(f'{model_name}_Confusion_ Matrix.png'))
    # if show:
    #     plt.show()
    #     plt.close()
    #     return None
    # else:
    #     return fig











