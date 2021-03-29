import sys
sys.path.insert(1, '..')

# her2bdl project name
from her2bdl import *

import tensorflow as tf
from tensorflow import keras
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
    print("\nImproved Predictive Distribution")
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
                batch_size=mc_dropout_batch_size
            )
            for z_i in z_batch#_arr
        ],
        repeat=repeat_it, number=3
    )
    display_profiling("Classifier", classifier_ipd_times)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Profiling and debuging tools to find bottlenecks in the architecture and other checks."
    )
    ap.add_argument("--gpu", dest="gpu", action="store_true", 
        help="Check GPU availability."
    )
    ap.add_argument("--summary", type=str, default=None,
        help="Display models summary, requiere a experiment configuration file."
    )
    ap.add_argument("--profiling",  type=str, default=None,
        help="Experiment time profiling, requiere a experiment configuration file."
    )

    args = vars(ap.parse_args())
    gpu = args["gpu"]
    summary_experiment = args["summary"]
    profiling_experiment = args["profiling"]

    if not any([gpu, summary_experiment, profiling_experiment]):
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

    print("Done")