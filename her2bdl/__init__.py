"""
Her2BDL
===

Uncertain Image Classificator with keras.

"""
# Check Environment Variables setted
import os
env_fails = []
if not os.environ.get("HER2BDL_HOME"): env_fails.append("HER2BDL_HOME")
if not os.environ.get("HER2BDL_DATASETS"): env_fails.append("HER2BDL_DATASETS")
if not os.environ.get("HER2BDL_EXPERIMENTS"): env_fails.append("HER2BDL_EXPERIMENTS")
if not os.environ.get("HER2BDL_EXTRAS"): env_fails.append("HER2BDL_EXTRAS")
assert len(env_fails) == 0, f"Required set environment variables: ({env_fails})"

from .data import *
from .models import *
from .visualization import *
from .tools import *

from .__version__ import __version__

