"""
Data generator to feed models
=============================

This modele define generator (keras.Sequential) to feed differents
models, with their defined input format.
"""

from tensorflow import keras
import numpy as np


__all__ = [
    'Sampler'
]

class Sampler:
    """
    Sampler

    Get patches from a WSI.
    """
    def __init__(self, patch_level, patch_size, patch_rotation_range=None, patch_zoom_range=None):
        # patch level
        self.patch_level = patch_level
        # patch size at given level
        self.patch_size = patch_size 
        # range for rotate patch (rotation invariant)
        self.patch_rotation_rage = patch_rotation_range # None means no apply rotation
        # range for zoom patch (scale invariant)
        self.patch_zoom_range = patch_zoom_range        # None means no apply zoom

    def get_from_size(self, wsi, probability_map, size, n=1):
        """
        Sampling n patches from WSI weighted by probability_map.    
        
        Parameters
        ==========
        wsi : `openslide.OpenSlide`
            Opened WSI.
        probability_map : `np.ndarray`
            Sampling weighted by this probability map for each pixel with this size.
        size : `tuple(int, int)`
            Size of thumbnail.
        n : `int`
            Number of samples. (default=1) 
        Returns
        =======
        `np.ndarray`
            Samples patches.
        """
        assert probability_map.shape[:2] == size, "size aren`t equal" 
        src = wsi.get_thumbnail(size)
        src_downsample = wsi.dimensions[0]/src.dimensions[0]
        dst_downsample = wsi.level_downsample[self.patch_level]

        channels = 0
        canvas = np.empty((n, self.patch_size[0], self.patch_size[1]))
        raise NotImplementedError


    def get_from_level(self, wsi, probability_map, level, n=1):
        """
        Sampling n patches from WSI weighted by probability_map.    
        
        Parameters
        ==========
        wsi : `openslide.OpenSlide`
            Opened WSI.
        probability_map : `np.ndarray`
            Sampling weighted by this probability map for each pixel at this level.
        level : `int
            Use level size for thumbnail.
        n : `int`
            Number of samples. (default=1) 
        Returns
        =======
        `np.ndarray`
            Samples patches.
        """
        size = wsi.level_dimensions[level]
        return self.get_from_size(wsi, probability_map, size, n=n)
 
    def set_patch_level(self, patch_level):
        'set output patch level.'
        self.patch_level = patch_level

    def set_patch_size(self, patch_size):
        'set output patch size.'
        self.patch_size = patch_size

class DataGenerator(keras.utils.Sequence):
    """
    DataGenerator

    For load dataset on the fly.
    """

    def __init__(self, dataset, batch_size,
                 preprocess_input_pipes={},
                 preprocess_output_pipes={},
                 sampler_parameters={},
                 shuffle=False
                 ):

        # Dataset
        self.dataset = dataset
    
        # Sampler
        self.sampler = Sampler(**sampler_parameters)

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