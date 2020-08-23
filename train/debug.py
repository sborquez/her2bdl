import sys

# AI project name
sys.path.insert(1, '..')
from her2bdl import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import timeline

from time import time
import logging


def check_gpus():
    cuda = tf.test.is_built_with_cuda()
    if cuda:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            #logging.info(f"Availables GPUs: {len(gpus)}" )
            print(f"Availables GPUs: {len(gpus)}" )
            for i, gpu in enumerate(gpus):
                #logging.info(f"Availables GPU {i}: {gpu}" )
                print(f"Availables GPU {i}: {gpu}" )
        else:
            #logging.info("Not availables GPUs.")
            print("Not availables GPUs.")
    else:
        #logging.info("Tensorflow is not built with CUDA")
        print("Tensorflow is not built with CUDA")

def profiling(model_path):
    raise NotImplementedError

def save_plot_model(model_path):
    raise NotImplementedError

if __name__ == "__main__":
    import argparse
    from .utils import *

    ap = argparse.ArgumentParser(description="Profiling and debuging tools to find bottlenecks in the architecture and other checks.")
    ap.add_argument("--gpu", dest="gpu", action="store_true", help="Check GPU availability.")

    args = vars(ap.parse_args())
    gpu = args["gpu"]

    if gpu:
        print("Running GPU Checking.")
        check_gpus()
    print("Done")