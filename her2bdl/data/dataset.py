"""
Dataset setup
=================================

This module handle data files and generate the datasets used
by the models.
"""

from . import INPUTS, TARGETS, GROUND_TRUTH_FILE, IMAGE_FILES 
import logging
from os.path import join, split, basename
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

__all__ = [
    'get_dataset', 
    'load_dataset', 'save_dataset', 'split_dataset',
    'aggregate_dataset', 'describe_dataset'
]


def get_dataset(source, include_ground_truth=True):
    """
    Get dataset from source.
    
    Parameters
    ==========
    source : `str`
        Path to dataset folder.
        datasets/
            * 01/
            * groundTruth.xlsx
    Returns
    =======
    `pd.DataFrame`
        Dataset .
    """
    images_folders = [ split(img)[0] for img in glob(join(source, '*/')) ]
    images_data = []
    for image_folder in images_folders:
        case_number = int(basename(image_folder))
        images_data.append({
            'source': source,
            'CaseNo': case_number,
            'image_her2': IMAGE_FILES[0].format(CaseNo=case_number),
            'image_he': IMAGE_FILES[1].format(CaseNo=case_number)
        })
    images_data = pd.DataFrame(images_data)
    if include_ground_truth:
        ground_truth_file = join(source, GROUND_TRUTH_FILE)
        ground_truth_data = pd.read_excel(ground_truth_file, skiprows=[0], usecols=[1,2])\
            .dropna()\
            .astype(int)
        dataset = pd.merge(ground_truth_data, images_data, on='CaseNo')
    else:
        images_data[TARGETS] = None
        dataset = images_data
    return dataset

def split_dataset(dataset, validation_ratio=0.1, test_ratio=None, seed=None):
    """Split dataset in train and validation sets (test optional) 
    using events and a given ratio. 

    This split conserve class balance.

    Parameters
    ==========
    dataset : `pd.DataFramer`
        Original dataset.
    validation_ratio : `float`
        Portion of images for validation dataset. (default=0.1)
    test_ratio : `float` or `None`
        Portion of images for test dataset. (default=`None`)
    seed : `int`or `None`
        Split by images
    Returns
    =======
    tuple of `pd.DataFrame`
        train, val or train, val, test
    """
    # Replicability
    if seed is not None:
        r = np.random.RandomState(seed)
    else:
        r = None

    # Splits by class/score
    train_all = []
    test_all = []
    validation_all = []
    for _, score_dataset in dataset.groupby('HeR2 SCORE'):
        # train/validation
        if test_ratio is None:
            train, validation = train_test_split(score_dataset, shuffle=True, random_state=r, test_size=validation_ratio)
            train_all.append(train)
            validation_all.append(validation)
        # train/validation/test
        else:
            train_and_val, test = train_test_split(score_dataset, shuffle=True, random_state=r, test_size=test_ratio)
            alpha = validation_ratio/(1. - test_ratio)
            train, validation = train_test_split(train_and_val, shuffle=True, random_state=r, test_size=alpha)
            train_all.append(train)
            validation_all.append(validation)
            test_all.append(test)

    # train/validation
    if test_ratio is None:
        train = pd.concat(train_all).sample(frac=1., random_state=r).reset_index().drop(columns='index')
        validation = pd.concat(validation_all).sample(frac=1., random_state=r).reset_index().drop(columns='index')
        return train, validation
    # train/validation/test
    else:
        train = pd.concat(train_all).sample(frac=1., random_state=r).reset_index().drop(columns='index')
        validation = pd.concat(validation_all).sample(frac=1., random_state=r).reset_index().drop(columns='index')
        test = pd.concat(test_all).sample(frac=1., random_state=r).reset_index().drop(columns='index')
        return train, validation, test

def load_dataset(dataset_filepath):
    """Load dataset.

    Parameters
    ----------
    dataset_filepath : `str`
        Path to dataset file. (.csv)
    Returns
    =======
    `pd.DataFrame`
        Loaded dataset    
    """
    dataset = pd.read_csv(dataset_filepath)
    return dataset

def save_dataset(dataset, output_folder, dataset_name):
    """Save Dataset.
    Parameters
    ----------
    dataset : `pd.Dataframe`
        Dataset.
    output_folder : `str`
        Folder containing saved file.
    dataset_name : `str`
        File name or dataset split (i.e. 'train').
    Returns
    =======
    `str`
        Dataset filepath 
    """
    filename = f"{dataset_name}.csv"
    filepath = join(output_folder, filename)
    dataset.to_csv(filepath, index=False)
    return filepath

"""
Utils Functions
===============
"""

def describe_dataset(dataset, include_targets=True):
    """
    Perform simple description of the dataset.

    Parameters
    ==========
    dataset : `pd.DataFrame`
        Dataset.
    include_targets: `bool`
        Include include_targets in description.
    """
    size = len(dataset)
    size_by_class = dataset[TARGETS[0]].value_counts()
    print('Dataset Info:')
    print(f'  size: {size}')
    print(f'  columns: {dataset.columns}' )
    if include_targets:
        print(f'  by class:')
        for score, size in size_by_class.items():
            print(f'    Score {score}: {size}')

def aggregate_dataset(dataset, replace_source=None):
    """
    Perform simple aggegation to dataset.

    Parameters
    ==========
    dataset : `for example pd.DataFrame`
        Dataset.
    replace_source : `str` or `None`
        Replace source columns.
    Returns
    =======
    `pd.DataFrame`
        Dataset with aggregate information.
    """
    if replace_source is not None:
        dataset.source = replace_source
    return dataset