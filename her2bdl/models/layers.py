"""
Custom Layers
=============

Collection of models custom layers.
"""

import tensorflow as tf
from tensorflow import keras


"""
Adapted from https://gist.github.com/raingo/a5808fe356b8da031837

Multi dimensional softmax,
refer to https://github.com/tensorflow/tensorflow/issues/210
compute softmax along the dimension of target
the native softmax only supports batch_size x dimension
"""
def softmax(target, axis, name=None):
    max_axis = tf.reduce_max(target, axis, keepdims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
    softmax = target_exp / normalize
    return softmax

import tensorflow_probability as tfp
MultivariateNormalTriL = tfp.layers.MultivariateNormalTriL