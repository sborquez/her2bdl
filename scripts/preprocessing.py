import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *

from pathlib import Path
import argparse
import logging


def preprocessing(dataset_filepath, output_folder=None, seed=None):
    """
    Apply ROI selection and save the result in a csv file.
    """
    train_dataset = load_dataset(dataset_filepath)

    if output_folder is None:
        output_folder = Path(dataset_filepath).parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Select ROIs from dataset.")
    ap.add_argument("-i", "--dataset", type=str, required=True,
                    help="Dataset csv file." )
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="Folder containing Roi selections files. (default dataset parent folder))")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random state seed.")
    args = vars(ap.parse_args())

    dataset_filepath = args["dataset"]
    output_folder = args["output"]
    seed = args["seed"]

    preprocessing(dataset_filepath, output_folder, seed=seed)


