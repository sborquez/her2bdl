"""
Loss Functions
==============

Collections of different Loss function for probability distributions
and distance matrix.
"""

import tensorflow as tf
from tensorflow import keras



def crossentropy_loss(dimensions=3, epsilon=1e-16):
    """Return the Cross Entropy loss function for multidimension probability map."""
    def loss(y_true, y_pred):
        """Cross entropy loss function."""
        axis = [-i for i in range(1, dimensions+1)]
        return K.sum(-K.log(y_pred + epsilon)*y_true, axis=axis)
    return loss