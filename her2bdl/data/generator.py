"""
Data generator to feed models
=============================

This modele define generator (keras.Sequential) to feed differents
models, with their defined input format.
"""

from tensorflow import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
import atexit
from .constants import *
from .wsi import *

__all__ = [
    'GridPatchGenerator',
    'MCPatchGenerator'
]

class GridPatchGenerator(keras.utils.Sequence):
    """
    Grid patch generator
    """

    def __init__(self, dataset, batch_size, patch_level, patch_size, 
                 patch_vertical_flip=False, patch_horizontal_flip=False, shuffle=True
                 ):
        # Generator parameters
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.shuffle = shuffle
        
        # Data augmentation parameters
        self.patch_vertical_flip = patch_vertical_flip
        self.patch_horizontal_flip = patch_horizontal_flip
    
        # Prepare dataset
        self.__setup__(dataset)
        self.on_epoch_end()
        
    
    def __setup__(self, dataset):
        'Setup dataset for patch extraction.'
        self.slides = {}
        # For each Slide
        patches = []
        total = dataset["CaseNo"].nunique()
        for case_no, df in tqdm(dataset.groupby("CaseNo"), total=total):
            self.slides[case_no] = open_slide(df.iloc[0]["slide_path"])
            # For each object in slide
            for _, row in df.iterrows():
                # Slide levels
                segmentation_level = row["segmentation_level"] # level where tissue was detected
                sampling_map_level = row["sampling_map_level"] # level where background is ignored
                patch_level        = self.patch_level          # level where patch are obtained 
                sampling_map       = np.load(row["sampling_map"]).astype(bool)
                # Downscale Patch size to sampling map level
                sampling_map_patch_size = level_scaler(self.patch_size, patch_level, sampling_map_level)
                # Find patches in sampling map scale with at least half of relevant pixels.
                sampling_map_patch_selector = strided_convolution(sampling_map, np.ones(sampling_map_patch_size), sampling_map_patch_size)
                sampling_map_patch_selector = np.argwhere(sampling_map_patch_selector > (np.prod(sampling_map_patch_size)*PATCH_RELEVANT_RATIO))
                # Scale indexes to sampling map level and and translate them with respect slide origin
                sampling_map_translation = level_scaler((row["min_row"], row["min_col"]), segmentation_level, sampling_map_level)
                sampling_map_patch_indexes = sampling_map_patch_size*sampling_map_patch_selector + sampling_map_translation
                # Upscale patches indexes from sampling map level to level 0 (for `read_region` method)
                patch_indexes = level_scaler(sampling_map_patch_indexes, sampling_map_level, 0, "numpy")
                # Build rows for generator`s dataset.
                new_patches = [{
                    "CaseNo": int(case_no),
                    "label": int(row["label"]),
                    TARGET: row[TARGET],
                    "row": patch[0],
                    "col": patch[1]
                } for patch in patch_indexes]
                patches += new_patches
        # Generator dataset with patches and scores only
        self.dataset = pd.DataFrame(patches)
        self.size = len(self.dataset)
        atexit.register(self.cleanup)

    def cleanup(self):
        'Close opened slides'
        for case_no in self.slides.keys():
            close_slide(self.slides[case_no])
            del self.slides[case_no]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.size)/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation__(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, list_indexes):
        'Generates data containing batch_size samples'
        # Initialization
        batch_rows = self.dataset.iloc[list_indexes]
        batch_patches = np.empty((len(list_indexes), *self.patch_size, 3), dtype=np.float64)
        batch_scores  = np.empty((len(list_indexes), len(TARGET_LABELS)))
        for i, (_, row) in enumerate(batch_rows.iterrows()):
            img = self.slides[int(row["CaseNo"])]
            patch = img.read_region((int(row["col"]), int(row["row"])), self.patch_level, self.patch_size)
            batch_patches[i] = np.array(patch)[:,:,:3] / 255.0
            batch_scores[i]  = TARGET_TO_ONEHOT[int(row[TARGET])]
        return batch_patches, batch_scores
    
class MCPatchGenerator(GridPatchGenerator):
    """
    Monte-Carlo Patch Generator 
    """

    def __init__(self, dataset, batch_size, patch_level, patch_size, samples_per_tissue=500,
                 patch_vertical_flip=False, patch_horizontal_flip=False, shuffle=True
                 ):
        # Take m samples from each tissue
        self.samples_per_tissue = samples_per_tissue
        super().__init__(dataset, batch_size, patch_level, patch_size, patch_vertical_flip, patch_horizontal_flip, shuffle)


    def __setup__(self, dataset):
        'Setup dataset for patch extraction.'
        # For each Slide
        self.slides = {}
        # Generator dataset with weights, patches and scores only
        self.dataset = pd.DataFrame()
        total = dataset["CaseNo"].nunique()
        self.patches = {}
        self.weights = {}
        for case_no, df in tqdm(dataset.groupby("CaseNo"), total=dataset["CaseNo"].nunique()):
            self.slides[case_no] = open_slide(df.iloc[0]["slide_path"])
            self.patches[case_no] = {}
            self.weights[case_no] = {}
            # For each object in slide
            for _, row in df.iterrows():
                # Slide levels
                segmentation_level = row["segmentation_level"] # level where tissue was detected
                sampling_map_level = row["sampling_map_level"] # level where background is ignored
                patch_level        = self.patch_level          # level where patch are obtained 
                sampling_map       = np.load(row["sampling_map"])
                # Sampler values and weights
                indexes = np.argwhere(sampling_map > 0)
                weights = sampling_map[indexes[:,0], indexes[:,1]]
                # Translate and scale to level 0
                sampling_map_translation = level_scaler((row["min_row"], row["min_col"]), segmentation_level, sampling_map_level)
                patch_indexes = level_scaler(indexes + sampling_map_translation, sampling_map_level, 0, "numpy")
                self.patches[case_no][row["label"]] = patch_indexes
                self.weights[case_no][row["label"]] = weights
                # Build rows for generator`s dataset.
                self.dataset = self.dataset.append({
                        "CaseNo": int(case_no),
                        "label" : int(row["label"]),
                        TARGET  : row[TARGET]
                }, ignore_index=True)

        self.size = len(self.dataset)*self.samples_per_tissue
        # How many pixels contain one pixel at sample level at patch level
        self.upsample_pixels_patch_level = level_scaler((1, 1), sampling_map_level, patch_level)
        self.upsample_pixels_level_0     = level_scaler((1, 1), patch_level, 0)
        atexit.register(self.cleanup)

    def on_epoch_end(self):
        'Updates samples after each epoch'
        self.indexes = np.empty((self.samples_per_tissue*len(self.dataset), 2), dtype=int)
        self.dataset_ref = np.empty((self.samples_per_tissue*len(self.dataset)), dtype=int)
        for i, (_, row) in enumerate(self.dataset.iterrows()):
            case_no = int(row["CaseNo"])
            label   = int(row["label"])
            score   = int(row[TARGET])
            # samples
            m       = self.samples_per_tissue
            indexes = self.patches[case_no][label]
            weights = self.weights[case_no][label]
            samples_indexes = indexes[np.random.choice(np.arange(len(indexes)), size=m)] 
            samples_fine_selection = np.random.randint((0,0), self.upsample_pixels_patch_level, size=(m, 2)) * self.upsample_pixels_level_0
            self.indexes[i*m: (i+1)*m] = samples_indexes + samples_fine_selection
            self.dataset_ref[i*m: (i+1)*m] = i
        if self.shuffle == True:
            shuffler = np.arange(len(self.indexes))
            np.random.shuffle(shuffler)
            self.indexes     = self.indexes[shuffler]
            self.dataset_ref = self.dataset_ref[shuffler]
        
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        dataset_ref = self.dataset_ref[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation__(indexes, dataset_ref)
        return X, y

    def __data_generation__(self, list_indexes, dataset_ref):
        'Generates data containing batch_size samples'
        # Initialization
        batch_rows = self.dataset.iloc[dataset_ref]
        batch_patches = np.empty((len(list_indexes), *self.patch_size, 3), dtype=np.float64)
        batch_scores  = np.empty((len(list_indexes), len(TARGET_LABELS)))
        for i, (_, row) in enumerate(batch_rows.iterrows()):
            case_no = int(row["CaseNo"])
            label   = int(row["label"])
            score   = int(row[TARGET])
            img = self.slides[case_no]
            location = (list_indexes[i][1], list_indexes[i][0])
            patch = img.read_region(location, self.patch_level, self.patch_size)
            batch_patches[i] = np.array(patch)[:,:,:3] / 255.0
            batch_scores[i]  = TARGET_TO_ONEHOT[score]
        return batch_patches, batch_scores
