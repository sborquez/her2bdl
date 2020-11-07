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
    'plot_predictive_distribution', 'plot_forward_pass_samples', 'plot_sample',
    'display_prediction', 'display_uncertainty'
]

#DEFAULT_PALETTE = "YlOrBr" # this is ideal for her2 representation
DEFAULT_PALETTE = "tab10"

def plot_predictive_distribution(predictive_distribution, target_names=None, axis=None):
    """
    Plot predictive distribution.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,4))
        axis = plt.gca()
    if target_names is not None:
        x = np.arange(len(target_names))
        ticks = [target_names[i] for i in x]
    else:
        x = list(range(len(predictive_distribution)))
        ticks = x
    #axis.bar(x, predictive_distribution)
    sns.barplot(x=x, y=predictive_distribution, ax=axis, palette=DEFAULT_PALETTE)
    axis.set_title("Predictive Distribution")
    # Style
    axis.set_xticks(x)
    axis.set_xticklabels(ticks, rotation=45)
    axis.set_xlabel("$C_k$")
    axis.set_ylabel("$p(y=C_k|x, \mathcal{D}_{train})$")
    axis.set_ylim([0, 1])
    #axis.legend()
    return axis

def plot_forward_pass_samples(prediction_samples, y_true=None, 
                              target_names=None, axis=None):
    """
    Plot samples distribution for each class.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,4))
        axis = plt.gca()
    if target_names is not None:
        labels = [target_names[i] for i in np.arange(len(target_names))]
        labels = [f"$C_{i}$: {l}"  for i,l in enumerate(labels)]
    else:
        labels = list(range(len(prediction_samples.T)))
        labels = [f"$C_{i}$: {i}"  for i in labels]
    if y_true is not None:
        labels[y_true] = f"{labels[y_true]} (true)"  
    df = pd.DataFrame(prediction_samples, columns=labels)
    g=sns.kdeplot(data=df, ax=axis, legend=True, fill=False, palette=DEFAULT_PALETTE)
    # Style
    axis.set_title("Samples Distribution by Class")
    axis.set_xlabel("$p(y=C_k|x, \hat{\omega}_t$)")
    axis.set_ylabel("Density")
    return axis

def plot_sample(x, y_true=None, y_pred=None, target_names=None, axis=None):
    if target_names is None:
        target_names = {}
    # Create new figure
    if axis is None:
        plt.figure(figsize=(6,6))
        axis = plt.gca()
    # Image
    axis.imshow(x)
    # Hide grid lines
    axis.grid(False)
    # Hide axes ticks
    axis.set_xticks([])
    axis.set_yticks([])
    # Style
    title = f""
    if y_true is not None:
        title = f"True - $C_{y_true}$: {target_names.get(y_true, y_true)}\n"
    if y_pred is not None:
        title += f"Predicted - $C_{y_pred}$: {target_names.get(y_pred, y_pred)}"        
    axis.set_title(title, y=-0.3)
    return axis

def display_uncertainty(x, y_pred, predictive_distribution, prediction_samples,
                        model_name=None, target_names=None,
                        y_true=None, uncertainty=None, 
                        show=False, save_to=None):
    """
    Display uncertainty for a sample x, plot predictive distribution and samples 
    class distributions.
    """
    # Create new Figure
    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(221)
    plot_sample(x, y_true=y_true, y_pred=y_pred, target_names=target_names, axis=ax1)

    ax2 = plt.subplot(222)
    plot_predictive_distribution(predictive_distribution, target_names=target_names, axis=ax2)
        
    # Uncertainty 
    if uncertainty is not None:
        textstr = '\n'.join([f"{metric}:{value:.3f}" for metric, value in uncertainty.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    
    ax3 = plt.subplot(212)
    plot_forward_pass_samples(prediction_samples, y_true=y_true, target_names=target_names, axis=ax3)
    plt.tight_layout()
    
    
    # Style
    title = f"Uncertainty - {model_name}" if model_name is not None else "Uncertainty"
    plt.suptitle(title)
    # Show or Save
    if save_to is not None:
        save_to = Path(save_to)
        fig.savefig(save_to.joinpath(f'{model_name}_Uncertainty.png'))
    if show:
        plt.show()
        plt.close()
        return None
    else:
        return fig
    
def display_prediction(x, y_pred, predictive_distribution, model_name=None, target_names=None, y_true=None, show=False, save_to=None):
    """
    Display the prediction of a input, the probability and the predicted point.
    If targets_values is not None, the target point is included in the figure.
    """
    # Create new Figure
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(121)
    plot_sample(x, y_true=y_true, y_pred=y_pred, target_names=target_names, axis=ax1)

    ax2 = plt.subplot(122)
    plot_predictive_distribution(predictive_distribution, target_names=target_names, axis=ax2)
    plt.tight_layout()
    # Style
    title = f"Prediction - {model_name}" if model_name is not None else "Prediction"
    plt.suptitle(title)
    # Show or Save
    if save_to is not None:
        save_to = Path(save_to)
        fig.savefig(save_to.joinpath(f'{model_name}_Prediction.png'))
    if show:
        plt.show()
        plt.close()
        return None
    else:
        return fig











