"""
Exploration Visualizations
======================

Generate plot for display dataset, inputs and targets.

Here you can find dataset exploration and samples visualizations.
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns; sns.set()


__all__ = [
    'display_class_distribution'
]


"""
Input Plots
=============
"""


"""
Target Plots
==============
"""


"""
Dataset Visualizations
=====================
"""
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


def display_class_distribution(dataset, target, target_labels, dataset_name=None, save_to=None):
    dataset_name = "Score Distribution" if dataset_name is None else f"Score Distribution - {dataset_name}"
    fig = plt.figure(figsize=(6,6))
    plt.title(dataset_name)
    classes = dataset[target].value_counts()
    labels = [target_labels[score] for score in classes.index]
    classes.plot.pie(labels=labels, autopct=make_autopct(classes), ax=plt.gca())
    # Show or Save
    if save_to is not None:
        fig.savefig(join(save_to, f'{dataset_name.replace(" ", "_")}.png'))
        plt.close(fig)
    else:
        plt.show()

