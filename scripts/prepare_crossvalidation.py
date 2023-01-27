import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *

import argparse
import logging
import pandas as pd
from pathlib import Path

def prepare_crossvalidation(source, kfolds=5, output=".", seed=None):
    dataset = aggregate_dataset(load_dataset(source))
    cv_splits = prepare_cv_splits(dataset, k=kfolds, seed=seed)
    output_folder = Path(output)
    for i, (tr, ts) in enumerate(cv_splits, start=1):
        save_dataset(tr, output_folder=output_folder, dataset_name=f"fold_{i}_training")
        save_dataset(ts, output_folder=output_folder, dataset_name=f"fold_{i}_validation")    

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare dataset for cross validation.")
    ap.add_argument("-i", "--source", type=str, required=True,
                    help="Dataset csv filepath.")
    ap.add_argument("-k", "--kfolds", type=int, default=5,
                    help="Number of subdivisions.")
    ap.add_argument("-o", "--output", type=str, default=".", 
                    help="Ouput folder.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random state seed.")
    kwargs = vars(ap.parse_args())

    datasets_paths =  prepare_crossvalidation(**kwargs)