"""
Data generator to feed models
=============================

This modele define generator (keras.Sequential) to feed differents
models, with their defined input format.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
import atexit
import cv2
from .constants import *
from .wsi import *
from .dataset import *

__all__ = [
    'get_generator_from_wsi', 'GridPatchGenerator', 'MCPatchGenerator', 
    'get_generators_from_tf_Dataset', 'get_generators_from_directory', 'generator_to_tf_Dataset'
]


def generator_to_tf_Dataset(generator, img_height, img_width):
    """
    Get tf.data.Dataset from generator.
    """
    def callable_iterator(generator):
        for img_batch, targets_batch in generator:
            yield img_batch, targets_batch
    num_classes = generator.num_classes 
    output_shape = ([None, img_height, img_width, 3], [None, num_classes])
    dataset = tf.data.Dataset.from_generator(
        lambda: callable_iterator(generator),
        output_types=(tf.float32, tf.float32), 
        output_shapes=output_shape
    )
    return dataset

def get_generators_from_tf_Dataset(dataset_target, input_shape, batch_size, 
                                   num_classes=None, label_mode="categorical", 
                                   validation_split=None, preprocessing={},
                                   data_dir=None):
    """
    Sources:
        * simple: https://www.tensorflow.org/datasets/catalog/mnist
        * binary: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs
        * multiclass: https://www.tensorflow.org/datasets/catalog/stanford_dogs
    """
    img_height, img_width = input_shape[:2]
    def get_mapper(img_height, img_width, to_rgb, num_classes, label_mode="categorical", rescale=None):
        @tf.autograph.experimental.do_not_convert
        def mapper(image, label):
            image = tf.image.resize(image, (img_height, img_width))
            if to_rgb:
                image = tf.image.grayscale_to_rgb(image)
            if rescale is not None:
                image = image * rescale
            if num_classes is not None:
                if label_mode == "categorical":
                    label = tf.one_hot(label, num_classes)
                else:
                    raise ValueError(f"Unkown label_mode: {label_mode}")
            return image, label
        return mapper

    assert (dataset_target in ["simple", "binary", "multiclass", "mnist"]),\
     "Invalid dataset_target."
    if dataset_target == "simple":
        to_rgb = True
        num_classes = 10
        dataset = 'mnist'
    elif dataset_target == "mnist":
        to_rgb = True
        num_classes = 10
        dataset = 'mnist'
    elif dataset_target == "binary":
        to_rgb = False
        num_classes = 2
        dataset = "cats_vs_dogs"
    elif dataset_target == "multiclass":
        to_rgb = False
        raise NotImplementedError("find num_classes for dogs")
        num_classes = None
        dataset = "stanford_dogs"

    # preprocessing
    rescale = preprocessing.get("rescale", None)

    # load and split datasets
    if validation_split is not None:
        if isinstance(validation_split, str):
            split = [f'train[:{2*batch_size}]', f'train[:{2*batch_size}]']
        elif isinstance(validation_split, float):
            #TODO: add split
            #validation_split
            split_ = int(10*validation_split)
            split = [f'train[:{100 - split_}%]', f'train[-{split_}%:]']
        train_ds, validation_ds = tfds.load(dataset, split=split, as_supervised=True, shuffle_files=True, batch_size=batch_size, data_dir=data_dir)
        train_dataset = train_ds.map(get_mapper(img_height, img_width, to_rgb, num_classes, label_mode=label_mode), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        steps_per_epoch = len(train_dataset)
        val_dataset = validation_ds.map(get_mapper(img_height, img_width, to_rgb, num_classes, label_mode=label_mode), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_steps = len(val_dataset)
        return (train_dataset, steps_per_epoch), (val_dataset, validation_steps)
    else:
        split = f'train[:{2*batch_size}]' if dataset_target == "simple" else "train" 
        ds = tfds.load(dataset, split=split, as_supervised=True, shuffle_files=True, batch_size=batch_size, data_dir=data_dir)
        steps_per_epoch = ds.map(get_mapper(img_height, img_width, to_rgb, num_classes, label_mode=label_mode), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        steps_per_epoch = len(steps_per_epoch)
        return (train_dataset, steps_per_epoch)


def get_generators_from_directory(data_directory, input_shape, batch_size, num_classes=None, label_mode="categorical", validation_split=None, preprocessing={}):
    rescale = preprocessing.get("rescale", None)
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=validation_split, 
        rescale=rescale
    )

    img_height, img_width = input_shape[:2]
    train_generator = image_generator.flow_from_directory(
        directory=str(data_directory),
        label_mode=label_mode,
        batch_size=batch_size,
        shuffle=True,
        target_size=(img_height, img_width),
        subset='training' if validation_split is not None else None
    )
    train_dataset = generator_to_tf_Dataset(train_generator, img_height, img_width)
    steps_per_epoch = len(train_generator)
    if validation_split is not None:
        val_generator = image_generator.flow_from_directory(
            directory=str(data_directory),
            label_mode=label_mode,
            batch_size=batch_size,
            shuffle=False,
            target_size=(img_height, img_width),
            subset='validation'
        )
        val_dataset  = generator_to_tf_Dataset(val_generator, img_height, img_width)
        validation_steps = len(val_generator)
        return (train_dataset, steps_per_epoch), (val_dataset, validation_steps)
    else:
        return (train_dataset, steps_per_epoch)


def get_generator_from_wsi(generator, input_shape, batch_size, num_classes=4,
                        validation_generator=None, label_mode='categorical', preprocessing={}):
    patch_size = input_shape[:2] # target_size
    img_height, img_width = patch_size
    generator_contructor = {
        "GridPatchGenerator": GridPatchGenerator,
        "MCPatchGenerator": MCPatchGenerator 
    }
    aggregate_dataset_parameters = preprocessing.get("aggregate_dataset_parameters", None)
    aggregate_dataset_parameters = aggregate_dataset_parameters or {}
    # Train generator
    generator_type = generator["generator"]
    generator_parameters = generator["generator_parameters"]
    generator_parameters["dataset"] = aggregate_dataset(
        load_dataset(generator_parameters["dataset"]),
        **aggregate_dataset_parameters
    )
    generator = generator_contructor[generator_type](
        batch_size=batch_size, 
        patch_size=patch_size, 
        label_mode=label_mode, 
        **generator_parameters
    )
    steps_per_epoch = int(generator.size // batch_size)
    dataset   = generator
    if validation_generator is not None:
        validation_generator_type = validation_generator["generator"]
        validation_generator_parameters = validation_generator["generator_parameters"]
        validation_generator_parameters["dataset"] = aggregate_dataset(
            load_dataset(validation_generator_parameters["dataset"]),
            **aggregate_dataset_parameters
        )
        validation_generator  = generator_contructor[validation_generator_type](
            batch_size=batch_size, 
            patch_size=patch_size, 
            label_mode=label_mode, 
            **validation_generator_parameters
        )
        validation_steps = int(validation_generator.size // batch_size)
        validation_dataset = validation_generator
        return (dataset, steps_per_epoch), (validation_dataset, validation_steps)
    else:
        return (dataset, steps_per_epoch)


class GridPatchGenerator(keras.utils.Sequence):
    """
    Grid patch generator
    """

    def __init__(self, dataset, batch_size, patch_level, patch_size, 
                 patch_vertical_flip=False, patch_horizontal_flip=False, 
                 label_mode="categorical", shuffle=True):
        # Generator parameters
        self.batch_size = batch_size 
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.shuffle = shuffle
        self.label_mode = label_mode
        self.patch_relevant_ratio = PATCH_RELEVANT_RATIO
        
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
                sampling_map_level = row["sampling_map_level"] # level with background mask
                patch_level        = self.patch_level          # level for patch extraction 
                # TODO: Check rows without sampling_maps
                sampling_map       = np.load(row["sampling_map"]).astype(bool) 
                # Downscale Patch size to sampling map level
                sampling_map_patch_size = level_scaler(self.patch_size, patch_level, sampling_map_level)
                # Find patches in sampling map scale with at least half of relevant pixels.
                if np.any(np.array(sampling_map.shape) < sampling_map_patch_size):
                    # patch_size is bigger than src tissue
                    diff = np.array(sampling_map_patch_size)-np.array(sampling_map.shape)
                    diff[diff<0] = 0
                    sampling_map = np.pad(sampling_map, pad_width=diff, mode="symmetric")
                sampling_map_patch_selector = strided_convolution(sampling_map, np.ones(sampling_map_patch_size), sampling_map_patch_size)
                sampling_map_patch_selector = np.argwhere(
                    # Number of relevant pixels > an area threshold
                    sampling_map_patch_selector > (np.prod(sampling_map_patch_size)*self.patch_relevant_ratio)
                )
                # Scale indexes to sampling map level and translate them with according to source slide
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
        self.num_classes = self.dataset[TARGET].nunique()
        if self.batch_size == -1:
            self.batch_size = len(self.dataset)
            self.size = len(self.dataset)
        else:
            self.size = self.batch_size * (len(self.dataset)//self.batch_size)
        atexit.register(self.cleanup)
 
    def cleanup(self):
        'Close opened slides'
        for case_no in self.slides.keys():
            close_slide(self.slides[case_no])
            #del self.slides[case_no]
        del self.slides

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.size//self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation__(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.indexes = np.random.choice(
                np.arange(len(self.dataset)), size=self.size, replace=False
            )
        else: #ignore last rows
            self.indexes = np.arange(self.size)
        if self.patch_vertical_flip:
            self.vertical_flips = np.random.choice((-1, 1), size=self.size)
        else:
            self.vertical_flips = np.ones(self.size, dtype=int)
        if self.patch_horizontal_flip:
            self.horizational_flips = np.random.choice((-1, 1), size=self.size)
        else:
            self.horizational_flips = np.ones(self.size, dtype=int)

    def __data_generation__(self, list_indexes):
        'Generates data containing batch_size samples'
        # Initialization
        batch_rows = self.dataset.iloc[list_indexes]
        batch_patches = np.empty((len(list_indexes), *self.patch_size, 3), dtype=np.float64)
        batch_scores  = np.empty((len(list_indexes), len(TARGET_LABELS)))
        for i, (_, row) in enumerate(batch_rows.iterrows()):
            case_no = int(row["CaseNo"])
            score   = int(row[TARGET])
            patch = self.load_patch(case_no, int(row["row"]), int(row["col"]))
            v_flip, h_flip = self.vertical_flips[i], self.horizational_flips[i]
            batch_patches[i] = np.array(patch)[::v_flip,::h_flip,:3]
            batch_scores[i]  = TARGET_TO_ONEHOT[score]
        return batch_patches, batch_scores

    def load_patch(self, case_no, row, col, patch_level=None, patch_size=None):
        patch_level = patch_level or self.patch_level
        patch_size = patch_size or self.patch_size
        img = self.slides[case_no]
        patch = img.read_region((col, row), patch_level, patch_size)    
        return patch

class MCPatchGenerator(GridPatchGenerator):
    """
    Monte-Carlo Patch Generator 
    """
    def __init__(self, dataset, batch_size, patch_level, patch_size, 
                 samples_per_tissue=100, patch_vertical_flip=True, 
                 patch_horizontal_flip=True, label_mode="categorical", 
                 shuffle=True):
        # Take m samples from each tissue
        self.samples_per_tissue = int(batch_size * (samples_per_tissue//batch_size))
        super().__init__(
            dataset, batch_size, patch_level, patch_size, patch_vertical_flip,
            patch_horizontal_flip, label_mode, shuffle
        )


    def __setup__(self, dataset):
        'Setup dataset for patch extraction.'
        # For each Slide
        self.slides = {}
        # Generator dataset with weights, patches and scores only
        self.dataset = pd.DataFrame()
        total = dataset["CaseNo"].nunique()
        self.patches = {}
        self.weights = {}
        # Move TODO: Check this hand selected values
        kernel = np.zeros((800, 800))
        kernel[400:, 400:] = 1/(400*400)
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
                weights = cv2.filter2D(sampling_map, -1, kernel)[indexes[:,0], indexes[:,1]]
                weights_sum = weights.sum()
                if weights_sum > 0:
                    weights /= weights.sum()
                else:
                    weights[:] = 1.0/len(weights)
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
        self.num_classes = self.dataset[TARGET].nunique()
        # How many pixels contain one pixel at sample level at patch level
        self.upsample_pixels_patch_level = level_scaler((1, 1), sampling_map_level, patch_level)
        self.upsample_pixels_level_0     = level_scaler((1, 1), patch_level, 0)
        atexit.register(self.cleanup)


    def on_epoch_end(self):
        'Updates samples after each epoch'
        self.size = len(self.dataset)*self.samples_per_tissue
        self.indexes = np.empty((self.size, 2), dtype=int)
        self.dataset_ref = np.empty((self.size), dtype=int)
        for i, (_, row) in enumerate(self.dataset.iterrows()):
            case_no = int(row["CaseNo"])
            label   = int(row["label"])
            score   = int(row[TARGET])
            # samples
            m       = self.samples_per_tissue
            indexes = self.patches[case_no][label] # index in wsi at level 0
            weights = self.weights[case_no][label] # 
            random_selector = np.random.choice(np.arange(len(indexes)), size=m, p=weights)
            samples_indexes = indexes[random_selector]
            if self.upsample_pixels_patch_level != (0, 0): # is higher resolution
                samples_fine_selection = np.random.randint((0,0), self.upsample_pixels_patch_level, size=(m, 2)) * self.upsample_pixels_level_0
                samples_indexes = samples_indexes + samples_fine_selection
            self.indexes[i*m: (i+1)*m] = samples_indexes
            self.dataset_ref[i*m: (i+1)*m] = i
        if self.shuffle:
            shuffler = np.arange(len(self.indexes))
            np.random.shuffle(shuffler)
            self.indexes     = self.indexes[shuffler]
            self.dataset_ref = self.dataset_ref[shuffler]
        if self.patch_vertical_flip:
            self.vertical_flips = np.random.choice((-1, 1), size=self.size)
        else:
            self.vertical_flips = np.ones(self.size, dtype=int)
        if self.patch_horizontal_flip:
            self.horizational_flips = np.random.choice((-1, 1), size=self.size)
        else:
            self.horizational_flips = np.ones(self.size, dtype=int)


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
            score   = int(row[TARGET])
            patch = self.load_patch(case_no, list_indexes[i][0], list_indexes[i][1])
            v_flip, h_flip = self.vertical_flips[i], self.horizational_flips[i]
            batch_patches[i] = np.array(patch)[::v_flip,::h_flip,:3]
            batch_scores[i]  = TARGET_TO_ONEHOT[score]
        return batch_patches, batch_scores
