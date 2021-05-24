import os
import gc as gc
import wandb as wandb
import numpy as np
import pandas as pd
from pathlib import Path as Path
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
from IPython.core.display import display, HTML

__all__ = [
    "display", "HTML",
    "gc", "wandb", "np", "pd", "Path", "plt", "tqdm",
    "reset_kernel", "os"
]

def reset_kernel(): 
    os._exit(00)


from time import time
from timeit import repeat
import logging

def display_profiling(name, times):
    mean = np.mean(times)
    std_err = np.std(times, ddof=1)
    print(f"{name}\n\tmean: {mean:.30f} [s]\n\tstd err: {std_err:.30f} [s]") 


def profiling(config, model, repeat_it=10):
    """
    Time profiling for mcdropout models.
    """
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
    #model = setup_model(input_shape, num_classes, **model_configuration, build=True)
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
