#%%
import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from her2bdl import setup_generators

#%%
def save_dataset(generator, output_folder, dataset_name):
    new_dataset = generator.dataset.copy()
    new_dataset.rename(columns = {'label':'TissueId'}, inplace=True)
    new_dataset["image"] = new_dataset.apply(
        lambda row: f"CaseNo_{row['CaseNo']}-TissueId_{row['TissueId']}-Score_{row['HeR2 SCORE']}-row_{row['row']}-col_{row['col']}.png",
        axis=1
    )
    os.makedirs(Path(output_folder)/dataset_name)
    for i in tqdm(range(len(generator))):
        x_batch, _ = generator[i]
        image = x_batch[0]
        image_name = new_dataset.at[i, "image"]
        image_path = Path(output_folder) / dataset_name / image_name
        Image.fromarray(image.astype(np.uint8)).save(image_path)
    new_dataset.to_csv(Path(output_folder) / dataset_name / "scores.csv", index=False)

#%%
def save_patches(data_configuration, output_folder):
    print("Saving training dataset...")
    (train_dataset, steps_per_epoch), input_shape, num_classes, labels = setup_generators(
        test_dataset=False, batch_size=1, **data_configuration
    )
    save_dataset(train_dataset, output_folder, "training")
    print("Saving test dataset...")
    (test_dataset, steps_per_epoch), input_shape, num_classes, labels = setup_generators(
        test_dataset=True,  batch_size=1, **data_configuration
    )
    save_dataset(test_dataset, output_folder, "test")
    print("done")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Save patch images from warwick datasets.")
    ap.add_argument("-o", "--output", type=str, required=True, 
        help="Output folder.")
    args = vars(ap.parse_args()) 
    output_folder = args["output"]
    data_configuration = {
        "source": {
            "type": "wsi",
            "parameters": { 
                "train_generator": {
                    "generator": 'GridPatchGenerator',
                    "generator_parameters": {
                    "dataset": 'D:/sebas/Projects/her2bdl/train/datasets/train.csv',
                    "patch_level": 3,
                    "patch_vertical_flip": False,
                    "patch_horizontal_flip": False,
                    "shuffle": False
                    }
                },
                "test_generator": {
                    "generator": 'GridPatchGenerator',
                    "generator_parameters": {
                    "dataset": 'D:/sebas/Projects/her2bdl/train/datasets/test.csv',
                    "patch_level": 3,
                    "patch_vertical_flip": False,
                    "patch_horizontal_flip": False,
                    "shuffle": False
                    }
                }
            }
        },
        "img_height": 300,
        "img_width": 300,
        "img_channels": 3,
        "preprocessing": {
            "rescale": None,
            "aggregate_dataset_parameters": None 
        },
        "num_classes": 4,
        "label_mode": "categorical",  
        "labels": ['0', '1+', '2+', '3+']
    }
    save_patches(data_configuration, output_folder)