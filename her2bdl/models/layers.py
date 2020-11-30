"""
Custom Layers
=============

Collection of models custom layers.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage.color import hed_from_rgb

__all__ = [
    'Separate_HED_stains'
]

class Separate_HED_stains(keras.layers.Layer):
    """
    Color descomposition of Her2 images in Haematoxylin-Eosin-DAB colorspace
    
    This layer takes input images of any shape, and the input data
    should range [0, 255]. Output is normalized to range [0, 1].
    
    This is a Keras implementation of the scikit-image `separate_stains` method.
    
    Haematoxylin-Eosin-DAB colorspace
    From original Ruifrok's paper: A. C. Ruifrok and D. A. Johnston,
    "Quantification of histochemical staining by color deconvolution.,"
    Analytical and quantitative cytology and histology / the International
    Academy of Cytology [and] American Society of Cytology, vol. 23, no. 4,
    pp. 291-9, Aug. 2001.
    
    """
    def __init__(self, **kwargs):
        super(Separate_HED_stains, self).__init__(**kwargs)
        self.hed_kernel = tf.constant(
            hed_from_rgb,
            shape=[1, 1, 3, 3],
            dtype=np.float32

        )
        self.log_adjust = tf.constant(np.log(10), dtype=np.float32)
        self.min_values = tf.constant(np.array([
             -0.69499301910400390625,
            -0.093443386256694793701171875,
            -0.542588651180267333984375
        ]), shape=[1, 1, 1, 3], dtype=np.float32)
        self.max_values = tf.constant(np.array([
            -0.2467001974582672119140625,
            0.36841285228729248046875,
            -0.143702208995819091796875
        ]), shape=[1, 1, 1, 3], dtype=np.float32)

    def call(self, inputs):
        x = inputs/255. + 2. # avoiding log artifacts
        # used to compensate the sum above
        x = tf.nn.conv2d(-tf.math.log(x)/self.log_adjust, self.hed_kernel, 1, "VALID")
        x = (x - self.min_values)/(self.max_values - self.min_values)
        return x
