"""
Inspect Visualizations
======================

Generate plot inspect the blackbox. 

Here you can find weights distributios, activations plots and model plot
"""

import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

__all__ = [
    'plot_predictive_distribution', 'plot_forward_pass_samples', 'plot_sample',
    'display_prediction', 'display_uncertainty', 'display_uncertainty_by_class',
    'display_map'
]

#DEFAULT_PALETTE = "YlOrBr" # this is ideal for her2 representation
DEFAULT_PALETTE = "tab10"

def plot_predictive_distribution(predictive_distribution, labels=None, axis=None):
    """
    Plot predictive distribution.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,4))
        axis = plt.gca()
    if labels is not None:
        x = np.arange(len(labels))
        ticks = labels
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
                              labels=None, axis=None):
    """
    Plot samples distribution for each class.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,4))
        axis = plt.gca()
    if labels is not None:
        labels_ = [labels[i] for i in np.arange(len(labels))]
        labels_ = [f"$C_{i}$: {l}"  for i,l in enumerate(labels_)]
    else:
        labels_ = list(range(len(prediction_samples.T)))
        labels_ = [f"$C_{i}$: {i}"  for i in labels_]
    if y_true is not None:
        labels_[y_true] = f"{labels_[y_true]} (true)"  
    df = pd.DataFrame(prediction_samples, columns=labels_)
    last = len(df)
    df.loc[last] = [0.0]*len(labels_)
    df.loc[last+1] = [1.0]*len(labels_)
    g=sns.histplot(data=df, ax=axis, legend=True, stat="density", fill=True, 
                   alpha=0.4, palette=DEFAULT_PALETTE, bins=11, element="poly")    
    # Style
    axis.set_title("Stochastic Forward Passes Distribution by Class")
    axis.set_xlabel("$p(y=C_k|x, \hat{\omega}_t$)")
    axis.set_xlim([0, 1])
    axis.set_ylabel("Density")
    return axis

def plot_sample(x, y_true=None, y_pred=None, labels=None, axis=None):
    """
    Plot input sample.
    """
    # Create new figure
    x = x.astype(np.float32)
    if axis is None:
        plt.figure(figsize=(6,6))
        axis = plt.gca()
    if x.max() > 1:
        x /= 255    # Image
    axis.imshow(x)
    # Hide grid lines
    axis.grid(False)
    # Hide axes ticks
    axis.set_xticks([])
    axis.set_yticks([])
    # Style
    title = f""
    if y_true is not None:
        label_ = y_true if labels is None else labels[y_true]
        title = f"True - $C_{y_true}$: {label_}\n"
    if y_pred is not None:
        label_ = y_pred if labels is None else labels[y_pred]
        title += f"Predicted - $C_{y_pred}$: {label_}"        
    axis.set_title(title, y=-0.3)
    return axis

def display_uncertainty(x, y_pred, predictive_distribution, prediction_samples=None,
                        model_name=None, labels=None, y_true=None, 
                        uncertainty=None, show=False, save_to=None):
    """
    Display uncertainty for a sample x, plot predictive distribution and samples 
    class distributions.

    Parameters
    ----------
    x : `np.ndarray`
        Input image (width, height, channels).
    y_pred :  `int`
        Predicted label.
    predictive_distribution : `np.ndarray`
        Predicted distribution (n_classes)
    prediction_samples : `np.ndarray` or `None`
        Predicted samples (samples, n_classes). Ignore second row plot if is `None`.
    model_name : `str`
        Model name.
    labels : `list`
        Label names ordered by index.
    y_true : `int`
        True label.
    uncertainty : `pd.Serie`
        ...
    show    : `bool`
        Show and close plot.
    save_to : `str` or `None`
        If is not `None`, save figure into filepath.
    Return
    ------
        `matplotlib.figure.Figure`.
    """
    # Create new Figure
    if prediction_samples is None:
        ax1_subplot = 121
        ax2_subplot = 122
        fig = plt.figure(figsize=(8, 4))
    else:
        ax1_subplot = 221
        ax2_subplot = 222
        fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(ax1_subplot)
    plot_sample(x, y_true=y_true, y_pred=y_pred, labels=labels, axis=ax1)
    ax2 = plt.subplot(ax2_subplot)
    plot_predictive_distribution(predictive_distribution, labels=labels, axis=ax2)
    # Uncertainty 
    if uncertainty is not None:
        textstr = '\n'.join([f"{metric}:{value:.3f}" for metric, value in uncertainty.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10, 
                verticalalignment='top', bbox=props)
    if prediction_samples is not None:
        ax3 = plt.subplot(212)
        plot_forward_pass_samples(prediction_samples, y_true=y_true, labels=labels, axis=ax3)
    # Style
    title = f"Uncertainty\n{model_name}" if model_name is not None else "Uncertainty"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

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
    
def display_prediction(x, y_pred, predictive_distribution, model_name=None, 
                       labels=None, y_true=None, show=False, save_to=None):
    """
    Display the prediction of a input, the probability and the predicted point.
    If targets_values is not None, the target point is included in the figure.
    
    Parameters
    ----------
    x : `np.ndarray`
        Input image (width, height, channels).
    y_pred :  `int`
        Predicted label.
    predictive_distribution : `np.ndarray`
        Predicted distribution (n_classes)
    prediction_samples : `np.ndarray`
        Predicted samples (samples, n_classes)
    model_name : `str`
        Model name.
    labels : `list`
        Label names ordered by index.
    y_true : `int`
        True label.
    show    : `bool`
        Show and close plot.
    save_to : `str` or `None`
        If is not `None`, save figure into filepath.
    Return
    ------
        `matplotlib.figure.Figure`.
    """
    # Create new Figure
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(121)
    plot_sample(x, y_true=y_true, y_pred=y_pred, labels=labels, axis=ax1)

    ax2 = plt.subplot(122)
    plot_predictive_distribution(predictive_distribution, labels=labels, axis=ax2)
    # Style
    title = f"Prediction\n{model_name}" if model_name is not None else "Prediction"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

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

def plot_uncertainty_by_class(y_true, uncertainty, metric="predictive entropy",
                     labels=None, axis=None):
    """
    Plot uncertainty boxplot by class.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(6,6))
        axis = plt.gca()
    new_df = uncertainty.copy()
    new_df["class"] =  y_true if labels is None else [labels[yi] for yi in y_true]
    sns.boxplot(data=new_df, x="class", y=metric)
    # Style
    axis.set_xlabel("$C_k$")
    title = f"{metric} by class"    
    return axis

def display_uncertainty_by_class(y_true, uncertainty, metric="predictive entropy",
                        model_name=None, labels=None, show=False, save_to=None):
    """
    Display uncertainty boxplot by class for a given metric.
    
    Parameters
    ----------
    y_true : `int`
        True label.
    uncertainty : `pd.Serie`
        ...
    metric : `str`
        Uncertainty metric.
    model_name : `str`
        Model name.
    labels : `list`
        Label names ordered by index.
    show    : `bool`
        Show and close plot.
    save_to : `str` or `None`
        If is not `None`, save figure into filepath.
    Return
    ------
        `matplotlib.figure.Figure`.
    """
    # Create new Figure
    fig = plt.figure(figsize=(8, 4))
    # Plot
    plot_uncertainty_by_class(y_true, uncertainty, 
            metric=metric, labels=labels, axis=plt.gca())
    # Style
    # Show or Save
    if save_to is not None:
        save_to = Path(save_to)
        fig.savefig(save_to.joinpath(f'{model_name}_{metric}_by_class.png'))
    if show:
        plt.show()
        plt.close()
        return None
    else:
        return fig

def display_map(image, heatmap=None, title=None, transparency=0.6, color_map='predictive entropy', show=False, save_to=None):
    fig = plt.figure(figsize=(8, 8))
    # Plot
    plt.imshow(image)
    if heatmap is not None:
        if color_map == "her2":
            color_map = ListedColormap(sns.color_palette(DEFAULT_PALETTE)[:4])
            plt.imshow(heatmap, alpha=transparency, cmap=color_map, vmin=-0.5, vmax=3.5)
            plt.colorbar(ticks=[0, 1, 2, 3], label="score")
        elif color_map == "mutual information":
            plt.imshow(heatmap, alpha=transparency, cmap="jet", vmin=0, vmax=1.5/3) #~np.log(4)
            plt.colorbar(label="uncertainty")
        elif color_map == 'predictive entropy':
            plt.imshow(heatmap, alpha=transparency, cmap="jet", vmin=0, vmax=1.5) #~np.log(4)
            plt.colorbar(label="uncertainty")
        else:
            plt.imshow(heatmap, alpha=transparency, cmap="jet", vmin=0, vmax=1.05) #~np.log(4)
            plt.colorbar(label="uncertainty")
    # Style
    if title is not None: plt.suptitle(title, fontsize=16)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # Show or Save
    if save_to is not None:
        save_to = Path(save_to)
        fig.savefig(save_to.joinpath(f'map.png'))
    if show:
        plt.show()
        plt.close()
        return None
    else:
        return fig








