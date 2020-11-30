"""
Models
======

Custom models, meta-models, layers and loss funtions.

"""

import tensorflow as tf

"""
Custom Loss
============
"""
from .loss import *
LOSS = {
    "CategoricalCrossentropy": tf.keras.losses.CategoricalCrossentropy
}


"""
Custom Layers
==============
"""
from .layers import *
LAYERS = {
    "Separate_HED_stains": Separate_HED_stains
}


"""
Custom Objects
==========
"""
CUSTOM_OBJECTS = {
    # LOSSES
    **LOSS,
    # LAYERS
    **LAYERS
}


"""
Optimizers
==========
"""
from tensorflow.keras.optimizers import (Adam, SGD, RMSprop)
OPTIMIZERS = {
    "adam"    : Adam,
    "sgd"     : SGD,
    "rmsprop" : RMSprop  
}


"""
Models and Aggregations
==========
"""
from .mcdropout import *

MODELS = {
    "SimpleClassifierMCDropout" : SimpleClassifierMCDropout,
    "EfficientNetMCDropout": EfficientNetMCDropout,
    "HEDConvClassifierMCDropout": HEDConvClassifierMCDropout

}
AGGREGATION_METHODS = {
}
