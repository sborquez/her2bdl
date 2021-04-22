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
from tensorflow.keras.optimizers import (Adam, SGD, RMSprop, Adadelta)
OPTIMIZERS = {
    "adam"    : Adam,
    "sgd"     : SGD,
    "rmsprop" : RMSprop,
    "adadelta": Adadelta
}


"""
Models and Aggregations
==========
"""
from .mcdropout import *
from .uncertainty import *
from .metrics import *

MODELS = {
    "SimpleClassifierMCDropout" : SimpleClassifierMCDropout,
    "EfficientNetMCDropout": EfficientNetMCDropout,
    "HEDConvClassifierMCDropout": HEDConvClassifierMCDropout,
    "RGBConvClassifierMCDropout": RGBConvClassifierMCDropout

}
AGGREGATION_METHODS = {
}
