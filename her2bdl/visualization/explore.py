"""
Exploration Visualizations
======================

Generate plot for display dataset, inputs and targets.

Here you can find dataset exploration and samples visualizations.
"""
from os import path
from pathlib import Path
from IPython.core.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns; sns.set()
from .. import IMAGE_FILES, IMAGE_IHC, COLORS

__all__ = [
    'display_WSI_and_metadata',
    'display_class_distribution',
    'display_wsi_sizes_distribution'
]

def open_slide(filename):
  """
  Open a whole-slide image (*.svs, etc).

  Args:
    filename: Name of the slide file.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  import openslide
  try:
    slide = openslide.open_slide(filename)
  except openslide.OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide

def close_wsi(wsi):
    wsi.close()

def open_wsi(source, caseno, image_type):
    slide_path = str( Path(source) / str(caseno).zfill(2) / image_type)
    wsi = open_slide(slide_path)
    return wsi

"""
Input Plots
=============
"""
def display_WSI_and_metadata(wsi):
    thumbnail = wsi.get_thumbnail(size=(800, 800))
    display(thumbnail)
    for i, v in enumerate(zip(wsi.level_downsamples, wsi.level_dimensions)):
        print(f"\tlevel [{i}]: Downsampling {v[0]} Size {v[1]}")
    return thumbnail

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
    figure_name = "Score Distribution" if dataset_name is None else f"Score Distribution - {dataset_name}"
    fig = plt.figure(figsize=(6,6))
    plt.title(figure_name)
    classes = dataset[target].value_counts()
    labels = [target_labels[score] for score in classes.index]
    colors = [COLORS[c] for c in classes.index]
    classes.plot.pie(labels=labels, autopct=make_autopct(classes), ax=plt.gca(), colors=colors)
    # Show or Save
    if save_to is not None:
        fig.savefig(path.join(save_to, f'{figure_name.replace(" ", "_")}.png'))
        plt.close(fig)
    else:
        plt.show()

def display_wsi_sizes_distribution(dataset, dataset_name=None, save_to=None):
    # Extract data points
    wsi_sizes = []
    for _, sample in dataset.iterrows():
        img = open_wsi(sample["source"], sample["CaseNo"],sample[IMAGE_IHC])
        wsi_sizes.append(img.dimensions)
        close_wsi(img)
    sizes = list(zip(*wsi_sizes))
    max_values = max(sizes[0]), max(sizes[1])
    # Create Figure
    figure_name = "WSI Sizes Distribution" if dataset_name is None else f"WSI Sizes Distribution - {dataset_name}"
    fig = plt.figure(figsize=(6,6))
    plt.title(figure_name)
    plt.scatter(x=sizes[0], y=sizes[1])
    plt.scatter(x=[0], y=[0], marker=",")
    plt.scatter(x=[max_values[0]], y=[max_values[1]], label=f"Max (w: {max_values[0]}, h: {max_values[1]})")
    #plt.gca().set_aspect('equals', adjustable='box')
    plt.axis('scaled')
    # style
    plt.xlim((0, max_values[0]))
    plt.ylim((0, max_values[1]))
    plt.xlabel("width")
    plt.ylabel("height")
    plt.legend()
    # Show or Save
    if save_to is not None:
        fig.savefig(path.join(save_to, f'{figure_name.replace(" ", "_")}.png'))
        plt.close(fig)
    else:
        plt.show()
  

