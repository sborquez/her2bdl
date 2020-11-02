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
Models and Ensemblers
==========
"""
from .model import *

MODELS = {
    "SimpleClassifierMCDropout" : SimpleClassifierMCDropout,
    "EfficentNetMCDropout": EfficentNetMCDropout
}
AGGREGATION_METHODS = {
}
