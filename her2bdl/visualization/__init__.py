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

from .metrics import *
from .explore import *
from .inspect import *
from .gui import *