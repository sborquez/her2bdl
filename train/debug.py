import sys
sys.path.insert(1, '..')

# her2bdl project name
from her2bdl import *

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K    
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import timeline

from time import time
from timeit import repeat
import logging


def inspect_model(config_file):
    """
    Model summary
    """
    # Load configuration and model
    config = load_config_file(config_file)

    ## disable plugins
    config["training"]["callbacks"]["enable_wandb"] = False 
    config["plugins"]["wandb"] = None

    ## hyperparameters
    epochs = config["training"]["epochs"]
    batch_size  = config["training"]["batch_size"]    
    input_shape = (
        config["data"]["img_width"],
        config["data"]["img_height"],
        config["data"]["img_channels"],
    )
    num_classes = config["data"]["num_classes"]
    model_configuration = config["model"]
    ## build model
    model = setup_model(input_shape, num_classes, **model_configuration, build=True)
    encoder, classifier = model.layers

    # Check model's parameters 
    model.summary()
    encoder.summary()
    classifier.summary()


def check_gpus():
    """
    Evaluate if tensorflow can access to the GPU.
    """
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


def display_profiling(name, times):
    mean = np.mean(times)
    std_err = np.std(times, ddof=1)
    print(f"{name}\n\tmean: {mean:.30f} [s]\n\tstd err: {std_err:.30f} [s]") 


def profiling(config_file, repeat_it=10):
    """
    Time profiling for mcdropout models.
    """
    # Load configuration and model
    config = load_config_file(config_file)
    ## disable plugins
    config["training"]["callbacks"]["enable_wandb"] = False 
    config["plugins"]["wandb"] = None
    ## hyperparameters
    batch_size  = config["training"]["batch_size"]
    input_shape = (
        config["data"]["img_width"],
        config["data"]["img_height"],
        config["data"]["img_channels"],
    )
    num_classes = config["data"]["num_classes"]
    model_configuration = config["model"]
    sample_size = model_configuration["uncertainty"]["sample_size"]
    mc_dropout_batch_size = model_configuration["uncertainty"]["mc_dropout_batch_size"]
    ## build model
    model = setup_model(input_shape, num_classes, **model_configuration, build=True)
    encoder, classifier = model.layers
    
    # Check model's parameters 
    model.summary()
    print("input shape", input_shape)
    print("batch size:", batch_size)
    print("Sample size:", sample_size)
    print("MC dropout batch size:", mc_dropout_batch_size)

    # Generate toy data
    X_batch = np.random.random((batch_size, *input_shape))
    z_batch = np.random.random((batch_size, encoder.output.shape[-1]))
    
    # Classic forward pass time
    print("Classic Fordward Pass")
    model_fp_times = repeat(
        lambda: model.predict(X_batch, batch_size=batch_size),
        repeat=repeat_it, number=3
    )
    display_profiling("Model", model_fp_times)
    encoder_fp_times = repeat(
        lambda: encoder.predict(X_batch, batch_size=batch_size),
        repeat=repeat_it, number=3
    )
    display_profiling("Encoder", encoder_fp_times)
    classifier_fp_times = repeat(
        lambda: classifier.predict(z_batch, batch_size=batch_size),
        repeat=repeat_it, number=3
    )
    display_profiling("Classifier", classifier_fp_times)

    # Vanilla predictive distribution
    model.predict(X_batch, batch_size=batch_size)  # Load any required 
    print("\nVanilla Predictive Distribution")
    model_vpd_times = repeat(
        lambda: [
            model.predict(X_batch, batch_size=batch_size) 
                for _ in range(mc_dropout_batch_size)
        ],
        repeat=repeat_it, number=3
    )
    display_profiling("Model", model_vpd_times)
    encoder_vpd_times = repeat(
        lambda: [
            encoder.predict(X_batch, batch_size=batch_size)
                for _ in range(mc_dropout_batch_size)
        ], 
        repeat=repeat_it, number=3
    )
    display_profiling("Encoder", encoder_vpd_times)
    classifier_vpd_times = repeat(
        lambda: [
            classifier.predict(z_batch, batch_size=batch_size)\
                for _ in range(mc_dropout_batch_size)
        ],
        repeat=repeat_it, number=3
    )
    display_profiling("Classifier", classifier_vpd_times)

    # Improved predictive distribution time
    model.predict_distribution(X_batch, batch_size=batch_size) # Load any required 
    print("\n2-Steps Predictive Distribution")
    model_ipd_times = repeat(
        lambda: model.predict_distribution(X_batch, batch_size=batch_size),
        repeat=repeat_it, number=3
    )
    display_profiling("Model", model_ipd_times)
    encoder_ipd_times = repeat(
        lambda: encoder.predict(X_batch, batch_size=batch_size),
        repeat=repeat_it, number=3
    )
    display_profiling("Encoder", encoder_ipd_times)
    classifier_ipd_times = repeat(
        lambda: [
            classifier.predict(
                np.tile(z_i, (sample_size, 1)),
                batch_size=sample_size#mc_dropout_batch_size*16
            )
            for z_i in z_batch#_arr
        ],
        repeat=repeat_it, number=3
    )
    display_profiling("Classifier", classifier_ipd_times)

def get_model_memory_usage(batch_size, model):
    """
    Model's memory usage in gigabytes.
    """
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

def memory_usage(config_file):
    """
    Display models' memory usage in gigabytes.
    """
    # Load configuration and model
    config = load_config_file(config_file)
    input_shape = (
        config["data"]["img_width"],
        config["data"]["img_height"],
        config["data"]["img_channels"],
    )
    num_classes = config["data"]["num_classes"]
    model_configuration = config["model"]
    # batch size
    batch_size  = config["training"]["batch_size"]
    mc_dropout_batch_size = model_configuration["uncertainty"]["mc_dropout_batch_size"]
    ## build model
    model = setup_model(input_shape, num_classes, **model_configuration, build=True)
    encoder, classifier = model.layers
    # Memory usage
    model_bytes = get_model_memory_usage(batch_size, model)
    encoder_bytes  = get_model_memory_usage(batch_size, encoder)
    classifier_bytes  = get_model_memory_usage(mc_dropout_batch_size, classifier)
    print(f"Model: {model_bytes}  GB.")
    print(f"Encoder: {encoder_bytes} GB.")
    print(f"Classifier: {classifier_bytes} GB.")


if __name__ == "__main__":
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_DISABLE_PTX_JIT'] = "1"
    import argparse
    ap = argparse.ArgumentParser(
        description="Profiling and debuging tools to find bottlenecks in the architecture and other checks."
    )
    ap.add_argument("--gpu", dest="gpu", action="store_true", 
        help="Check GPU availability."
    )
    ap.add_argument("--summary", type=str, default=None,
        help="Display models' summary, requiere a experiment configuration file."
    )
    ap.add_argument("--profiling",  type=str, default=None,
        help="Experiment time profiling, requiere a experiment configuration file."
    )
    ap.add_argument("--memory_usage",  type=str, default=None,
        help="Display model's memory usage, requiere a experiment configuration file."
    )

    args = vars(ap.parse_args())
    gpu = args["gpu"]
    summary_experiment = args["summary"]
    profiling_experiment = args["profiling"]
    memory_usage_experiment = args["memory_usage"]

    if not any([gpu, summary_experiment, profiling_experiment, memory_usage_experiment]):
        ap.print_help()

    if gpu:
        print("Running GPU Checking.")
        check_gpus()

    if profiling_experiment is not None:
        print("Running model profiling")
        profiling(profiling_experiment)

    if summary_experiment is not None:
        print("Model summaries")
        inspect_model(summary_experiment)

    if memory_usage_experiment is not None:
        print("Model summaries")
        memory_usage(memory_usage_experiment)
        
    print("Done")