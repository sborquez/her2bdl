import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *

import argparse
import logging
from tqdm import tqdm
from os import path


def prepare_dataset(source, output, split_validation=0.1, split_test=None, include_ground_truth=True, select_roi=False, seed=None):
    # 0. Get Dataset from source
    dataset = get_dataset(source, include_ground_truth=include_ground_truth)
        
    # 1. Select roi
    if select_roi:
        level = WSI_DEFAULT_LEVEL
        for _, sample in tqdm(dataset.iterrows(), total=len(dataset)):
            slide_filepath = path.join(sample["source"], str(sample["CaseNo"]).zfill(2), sample[IMAGE_IHC])
            slide = open_slide(slide_filepath)
            objects, labeled_segmentation = select_roi_manual(slide, level=level)
            save_objects(slide, sample["CaseNo"], IMAGE_IHC, objects, labeled_segmentation, output, level=level, mode="a")
            close_slide(slide)

    # 2. Fit Probability Sampling Maps

    # 3. Split train/validation/test
    if split_test is not None:
        train, val, test = split_dataset(dataset, validation_ratio=split_validation, test_ratio=split_test, seed=seed)
    else:
        train, val = split_dataset(dataset, validation_ratio=split_validation, seed=seed)

    # 4. Save
    train_csv = save_dataset(train, output, "train")
    validation_csv = save_dataset(val, output, "validation")
    if split_test is not None:
        test_csv = save_dataset(test, output, "test")
        return train_csv, validation_csv, test_csv
    else:
        return train_csv, validation_csv 

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare dataset files for generators.")
    ap.add_argument("-i", "--source", type=str, required=True,
                    help="Dataset folder.")
    ap.add_argument("-g", "--include_ground_thuth", dest="include_ground_truth", action='store_true',
                    help="Include Ground Truth.")
    ap.add_argument("-o", "--output", type=str, default=".", 
                    help="Ouput folder.")
    ap.add_argument("-s", "--split_validation", type=float, default=0.1,
                    help="Validation ratio for split data.")
    ap.add_argument("-S", "--split_test", type=float, default=None,
                    help="Test ratio for split data."),
    ap.add_argument("-r", "--roi", dest="select_roi", action="store_true",
                    help="Include manual roi selection step.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random state seed.")
    kwargs = vars(ap.parse_args())

    prepare_dataset(**kwargs)