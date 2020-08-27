import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *

import argparse
import logging


def prepare_dataset(source, output_folder, split_val=0.1, split_test=None, include_ground_truth=True, seed=None):
    # 1. Get Dataset from source
    dataset = get_dataset(source, include_ground_truth=include_ground_truth)
        
    # 2. Split train/validation/test
    if split_test is not None:
        train, val, test = split_dataset(dataset, validation_ratio=split_val, test_ratio=split_test, seed=seed)
    else:
        train, val = split_dataset(dataset, validation_ratio=split_val, seed=seed)

    # 3. Save
    train_csv = save_dataset(train, output_folder, "train")
    validation_csv = save_dataset(val, output_folder, "validation")
    if split_test is not None:
        test_csv = save_dataset(test, output_folder, "test")
        return train_csv, validation_csv, test_csv
    else:
        return train_csv, validation_csv 

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare dataset files for generators.")
    ap.add_argument("-i", "--source", type=str, required=True,
                    help="Dataset folder")
    ap.add_argument("-g", "--include_ground_thuth", dest="include_ground_truth", action='store_true',
                    help="Include Ground Truth.")
    ap.add_argument("-o", "--output", type=str, default=".", 
                    help="Ouput folder.")
    ap.add_argument("-s", "--split", type=float, default=0.1,
                    help="Validation ratio for split data.")
    ap.add_argument("-S", "--test", type=float, default=None,
                    help="Test ratio for split data.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random state seed.")
    args = vars(ap.parse_args())

    source = args["source"]
    include_ground_truth = args["include_ground_truth"]
    output_folder = args["output"]
    split_val = args["split"]
    split_test = args["test"]
    seed = args["seed"]

    prepare_dataset(source, output_folder, 
                    split_val=split_val, split_test=split_test, include_ground_truth=include_ground_truth, seed=seed)