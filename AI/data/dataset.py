"""
Dataset setup
=================================

This module handle data files and generate the datasets used
by the models.
"""

import logging

__all__ = [
    #'get_dataset', 
    #'load_dataset', 'save_dataset', 'split_dataset',
    #'filter_dataset', 'aggregate_dataset', 'describe_dataset'
]




def get_dataset():
    """
    Get dataset from a source. 
    """
    raise NotImplementedError


def split_dataset(dataset, validation_ratio=0.1):
    """Split dataset in train and validation sets using events and a given ratio. 
    """
    raise NotImplementedError

def load_dataset():
    """Load dataset.
    """
    raise NotImplementedError

def save_dataset(dataset):
    """Save Dataset.
    Parameters
    ----------
    dataset : `pd.Dataframe`
        Dataset.
    """
    raise NotImplementedError
    

"""
Utils Functions
===============
"""

def describe_dataset(dataset):
    """
    Perform simple description of the dataset.

    Parameters
    ==========
    dataset : `for example pd.DataFrame`
        Dataset.
    """
    raise NotImplementedError

def aggregate_dataset(dataset):
    """
    Perform simple aggegation to dataset.

    Parameters
    ==========
    dataset : `for example pd.DataFrame`
        Dataset.
    Returns
    =======
    `pd.DataFrame`
        Dataset with aggregate information.
    """
    raise NotImplementedError

def filter_dataset(dataset):
    """
    Select a subset from the dataset given some restrictions.

    Parameters
    ==========
    dataset : `for example pd.DataFrame`
        Dataset.
    Returns
    =======
     `pd.DataFrame`
        Filtered dataset.
    """
    raise NotImplementedError
