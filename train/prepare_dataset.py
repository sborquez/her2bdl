import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *

import argparse
import logging


def prepare_dataset(...):
    pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare dataset files for generators.")
    ap.add_argument("-o", "--output", type=str, default=".", 
                    help="Ouput folder.")
    ap.add_argument("-s", "--split", type=float, default=0.1,
                    help="Validation ratio for split data.")
    args = vars(ap.parse_args())

    output = args["output"]
    split = args["split"]

    prepare_ataset(...)