import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from AI import *

import logging
import time
from os import path


def train_model():
    pass

if __name__ == "__main__":
    import argparse
    from .utils import *

    ap = argparse.ArgumentParser(description="Train a model.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Configuration file for model/experiment.")
    ap.add_argument("-q", "--quiet", action='store_true', dest='quiet')
    ap.add_argument("-G", "--multi-gpu", action='store_true', dest='multi_gpu')
    args = vars(ap.parse_args()) 
    config_file = args["config"]
    quiet = args["quiet"]
    multi_gpu = args["multi_gpu"]
    
    print(f"Loading config from: {config_file}")
    experiment_config = load_config_file(config_file)

    model = train_model(...)