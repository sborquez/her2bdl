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
from .uncertainty import *
from .metrics import *

from .mcdropout import (
    SimpleClassifierMCDropout,
    EfficientNetMCDropout,
    HEDConvClassifierMCDropout,
    RGBConvClassifierMCDropout  
)
MODELS = {
    "SimpleClassifierMCDropout" : SimpleClassifierMCDropout,
    "HEDConvClassifierMCDropout": HEDConvClassifierMCDropout,
    "RGBConvClassifierMCDropout": RGBConvClassifierMCDropout,
    "EfficientNetMCDropout":      EfficientNetMCDropout
}

from .aggregation import (
    ThresholdAggregator,
    MixtureAggregator
)    
AGGREGATION_METHODS = {
    "ThresholdAggregator":  ThresholdAggregator,
    "MixtureAggregator":    MixtureAggregator
}
