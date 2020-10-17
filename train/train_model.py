import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *

import logging
import time
from os import path


def train_model(display=None):
    pass

if __name__ == "__main__":
    import argparse
    from .tools import *

    ap = argparse.ArgumentParser(description="Train a model.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Configuration file for model/experiment.")
    ap.add_argument("-q", "--quiet", action='store_true', dest='quiet')
    args = vars(ap.parse_args()) 
    config_file = args["config"]
    quiet = args["quiet"]
    
    print(f"Loading config from: {config_file}")
    experiment_config = load_config_file(config_file)

    model = train_model(display=GUIcmd(), **experiment_config)