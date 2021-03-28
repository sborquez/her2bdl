from nose.tools import raises, timed, eq_, ok_
from nose import with_setup
import tensorflow as tf


def test_tensorflow_gpu():
    """
    Tensorflow build with CUDA.
    """
    cuda = tf.test.is_built_with_cuda()
    ok_(cuda, "Tensorflow is not built with CUDA")


def test_tensorflow_gpu():
    """
    Tensorflow access to the GPU.
    """
    gpus = tf.config.list_physical_devices('GPU')
    ok_(len(gpus) > 0, "Not availables GPUs.")
    