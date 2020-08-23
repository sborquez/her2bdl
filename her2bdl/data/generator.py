"""
Data generator to feed models
=============================

This modele define generator (keras.Sequential) to feed differents
models, with their defined input format.
"""

from tensorflow import keras
import numpy as np


__all__ = [
    #'DataGenerator'
]

class DataGenerator(keras.utils.Sequence):
    """
    DataGenerator

    For load dataset on the fly.
    """

    def __init__(self, dataset, 
                 batch_size,
                 preprocess_input_pipes={},
                 preprocess_output_pipes={},
                 shuffle=False
                 ):

        # Dataset
        self.dataset = dataset
    
        # Input Normalization and stardarization 
        self.preprocess_input_pipes = preprocess_input_pipes

        # Output post-processing
        self.preprocess_output_pipes = preprocess_output_pipes

        # Generator parameters
        self.batch_size = batch_size
        self.size = len(self.dataset)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.size)
        # TODO: Add shuffle but without mix hdf5 files
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples'
        # Initialization
        raise NotImplementedError