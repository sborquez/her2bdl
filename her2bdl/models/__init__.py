"""
Models
======

Custom models, meta-models, layers and loss funtions.

"""



"""
Custom Loss
============
"""
from .loss import *
LOSS = {
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
    "loss": crossentropy_loss(), #dummy loss
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

MODELS = {
}

ENSEMBLERS = {
}



