import os
import gc as gc
import wandb as wandb
import numpy as np
import pandas as pd
from pathlib import Path as Path
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
from IPython.core.display import display, HTML

__all__ = [
    "display", "HTML",
    "gc", "wandb", "np", "pd", "Path", "plt", "tqdm",
    "reset_kernel"
]

def reset_kernel():
    os._exit(00)
