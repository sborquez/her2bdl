"""
Visualization
=============

Generate plots and show information.

GUI Class manages canvas (or windows) and user input.

`display_*` funtions generate a figure. These can be complex figures or independent
figure. 

`plot_*` functions generate individual and simples plots in an axis. This
type of function can be used by `display_*` to combine different figures.
"""

from .performance import *
from .explore import *
from .prediction import *
from .gui import *