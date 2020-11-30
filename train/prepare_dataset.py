import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *

import argparse
import logging
from os import path
import pandas as pd


def prepare_dataset(source, output, 
                    split_validation=0.1, split_test=None, include_ground_truth=True, 
                    select_roi=False, show_rois = True,
                    get_sampling_map=False, sampling_map_mode='uniform',
                    seed=None,
                    display=None
                    ):
    # 0. Get Dataset from source
    dataset = get_dataset(source, include_ground_truth=include_ground_truth)
        
    # 1. Select roi
    if select_roi:
        level = WSI_SEGMENTATION_LEVEL
        objects_df_rows = []
        if display is not None:
            iterator = display.progress_bar(dataset.iterrows(), total=len(dataset))
        else:
            iterator = dataset.iterrows()
        for _, sample in iterator:
            slide_filepath = path.join(sample["source"], str(sample["CaseNo"]).zfill(2), sample[IMAGE_IHC])

            slide = open_slide(slide_filepath)
            score = sample[TARGET]
            rois, labeled_segmentation = select_roi_manual( slide, level=level, 
                title=f"ROI Selection - HER2 Score {score} - ", display=display
            )
            segmentation_path = save_segmentation(sample["CaseNo"], labeled_segmentation, output)
            for selected, is_guess, label, centroid, bbox in rois:
                size = labeled_segmentation.shape[:2][::-1]
                centroid_row, centroid_col = centroid
                min_row,min_col,max_row,max_col = bbox
                new_object = {
                    "slide_path": slide_filepath,
                    "source": sample["source"],
                    "CaseNo": sample["CaseNo"],
                    "segmentation_level": level,
                    "segmentation_width": size[0],
                    "segmentation_height": size[1],
                    "selected": selected,
                    "is_guess": is_guess,
                    "label":    label,
                    "segmentation_path": segmentation_path,
                    "centroid_row": centroid_row,
                    "centroid_col": centroid_col,
                    "min_row": min_row,
                    "min_col": min_col,
                    "max_row": max_row,
                    "max_col": max_col,
                }
                objects_df_rows.append(new_object)
            close_slide(slide)
        objects = pd.DataFrame(objects_df_rows)
    else:
        objects_path = path.join(output, f"objects_{IMAGE_IHC}.csv")
        objects = load_objects(objects_path)

    # 2. Fit Probability Sampling Maps
    if get_sampling_map or select_roi:
        level = WSI_SAMPLING_MAP_LEVEL
        objects["sampling_map"] = np.nan
        objects["sampling_map_level"] = level
        sampling_maps_paths = []
        filtered_objects = objects["selected"]
        if display is not None:
            iterator = display.progress_bar(objects.loc[filtered_objects].iterrows(), total=objects["selected"].sum())
        else:
            iterator = objects.loc[filtered_objects].iterrows()
        for _, obj in iterator:
            slide_filepath = obj["slide_path"]
            slide = open_slide(slide_filepath)
            if show_rois:
                sampling_map = fit_sampling_map(slide, obj, level=level, mode=sampling_map_mode, display=display)
            sampling_map = fit_sampling_map(slide, obj, level=level, mode=sampling_map_mode, display=None)
            sampling_map_path = save_sampling_map(obj["CaseNo"], obj["label"], sampling_map, output)
            sampling_maps_paths.append(sampling_map_path)
            close_slide(slide)
        objects.loc[filtered_objects, "sampling_map"] = sampling_maps_paths

    # 3. Split train/validation/test
    if split_test is not None:
        train, val, test = split_dataset(dataset, validation_ratio=split_validation, test_ratio=split_test, seed=seed)
    else:
        train, val = split_dataset(dataset, validation_ratio=split_validation, seed=seed)

    # 4. Save
    if get_sampling_map or select_roi:
        save_objects(objects, IMAGE_IHC, output)

    train_csv = save_dataset(train, objects, output, "train")
    validation_csv = save_dataset(val, objects, output, "validation")
    if split_test is not None:
        test_csv = save_dataset(test, objects, output, "test")
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
    ap.add_argument("--show_rois", dest="show_rois", action="store_true",
                    help="Include manual roi selection step.")
    ap.add_argument("-m", "--get_sampling_map", dest="get_sampling_map", action="store_true",
                    help="Include sampling maps fit.")
    ap.add_argument("--sampling_map_mode", type=str, default='uniform',
                    help="Sampling map mode."),          
    ap.add_argument("--seed", type=int, default=None,
                    help="Random state seed.")
    kwargs = vars(ap.parse_args())

    if DEBUG:
        print("DEBUG Mode activated")

    datasets_paths =  prepare_dataset(display=GUIcmd(), **kwargs)