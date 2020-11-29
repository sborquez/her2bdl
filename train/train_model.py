import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *

import logging
import time
from os import path


def train_model(config, quiet=False, run_dir=".", display=None):
    # Seed
    seed = config["experiment"]["seed"]
    if seed is not None:
        pass #TODO: add seed

    # Experiment paths and indentifiers
    experiments_folder = config["experiment"]["experiments_folder"]
    experiment_name    = config["experiment"]["name"]
    experiment_folder  = path.join(experiments_folder, experiment_name)
    run_id             = config["experiment"]["run_id"]
    # Dataset
    source_type = config["data"]["source"]["type"]
    dataset_parameters = config["data"]["source"]["parameters"]
    input_shape = (
        config["data"]["img_height"], 
        config["data"]["img_width"],
        config["data"]["img_channels"]
    )
    preprocessing = config["data"]["preprocessing"]
    num_clasess = config["data"]["num_classes"]
    label_mode = config["data"]["label_mode"]
    labels = config["data"]["labels"]
    if labels == "HER2": labels = TARGET_LABELS_list
    batch_size  = config["training"]["batch_size"]
    validation_split = config["training"]["validation_split"]
    # Load train and validation generators
    if source_type == "tf_Dataset":
        print("Loading tf_Dataset generators:")
        train_, val_ = get_generators_from_tf_Dataset(
            **dataset_parameters, 
            num_classes=num_clasess, label_mode=label_mode,
            input_shape=input_shape, batch_size=batch_size, 
            validation_split=validation_split, preprocessing=preprocessing
        )
        (train_dataset, steps_per_epoch) = train_
        (val_dataset, validation_steps)  = val_
    elif source_type == "wsi":
        # ignore test dataset 
        __test_generator = dataset_parameters["test_generator"] 
        del dataset_parameters["test_generator"]
        # 
        print("Loading WSI generators:")
        train_, val_ = get_generator_from_wsi(
            **dataset_parameters, 
            num_classes=num_clasess, label_mode=label_mode,
            input_shape=input_shape, batch_size=batch_size,
            preprocessing=preprocessing
        )
        (train_dataset, steps_per_epoch) = train_
        (val_dataset, validation_steps)  = val_
        dataset_parameters["test_generator"] = __test_generator
        del __test_generator
    # elif source_type == "directory":
    #     # TODO: add get_generators_from_directory
    #     raise NotImplementedError
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    # Model architecture
    task  = config["model"]["task"]
    architecture = config["model"]["architecture"]
    if architecture not in MODELS: 
        raise ValueError(f"Unknown architecture: {architecture}")
    base_model = MODELS[architecture]
    model = base_model(
        input_shape, num_clasess, 
        **config["model"]["hyperparameters"], 
        **config["model"]["uncertainty"]
    )
    if config["model"]["weights"] is not None:
        weights = config["model"]["weights"]
        model.load_weights(weights)
    
    # Training parameters
    epochs = config["training"]["epochs"]
    batch_size  = config["training"]["batch_size"]
    validation_split = config["training"].get("validation_split", None)
    ## Loss
    loss_function    = config["training"]["loss"]["function"]
    loss_parameters  = config["training"]["loss"]["parameters"]
    loss_parameters  = loss_parameters or {}
    loss = LOSS[loss_function](**loss_parameters)
    ## Metrics
    metrics = config["evaluate"]["metrics"]
    ## Optimizer
    optimizer_name = config["training"]["optimizer"]["name"]
    optimizer_learning_rate = float(config["training"]["optimizer"]["learning_rate"]) # fix scientific notation parsed as str.
    optimizer_parameters = config["training"]["optimizer"].get("parameters", {})
    optimizer_parameters = optimizer_parameters or {}
    optimizer = OPTIMIZERS[optimizer_name](
        learning_rate=optimizer_learning_rate, 
        **optimizer_parameters
    )
    ## Callbacks
    enable_wandb = config["training"]["callbacks"]["enable_wandb"]
    earlystop = config["training"]["callbacks"]["earlystop"]
    experiment_tracker = config["training"]["callbacks"]["experiment_tracker"]
    checkpoints = config["training"]["callbacks"]["checkpoints"]
    callbacks = setup_callbacks(
         validation_data=val_dataset, 
         validation_steps=validation_steps,
         model_name=experiment_name,
         batch_size=batch_size,
         enable_wandb=enable_wandb,
         labels=labels,
         earlystop=earlystop,
         experiment_tracker=experiment_tracker,
         checkpoints=checkpoints,
         run_dir=run_dir
    )
    # Train
    model.compile(
        optimizer=optimizer,
        loss=loss,  
        metrics=metrics
    )
    history = model.fit(train_dataset, 
        verbose = 2 if quiet else 1,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset, 
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks
    )
    # Return final model
    return model

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train a model.")
    ap.add_argument("-c", "--config", type=str, required=True, 
        help="Configuration file for model/experiment.")
    ap.add_argument("--dryrun", action='store_true', 
        help="Run locally. Upload your results to wandb servers afterwards.")
    ap.add_argument("--quiet", action='store_true', 
        help="Disable progress bar.") 
    ap.add_argument("--disable_wandb", action='store_true', 
        help="Disable WandB for locally testing without Weight&Bias Callbacks.")
    ap.add_argument("--job", type=int, default=None, 
        help="Disable WandB for locally testing without Weight&Bias Callbacks.")
    args = vars(ap.parse_args()) 

    # Load experiment configuration
    config_file = args["config"]
    print(f"Loading config from: {config_file}")
    experiment_config = load_config_file(config_file)

    # Configure multiples runs
    job = args["job"]
    if job is not None:
        model_name = experiment_config["experiment"]["name"]
        job_sufix  = f"job {str(job).zfill(2)}"
        experiment_config["experiment"]["name"]= f"{model_name} {job_sufix}"
    # Setup experiments and pluggins
    if args["disable_wandb"]:
        experiment_config["training"]["callbacks"]["enable_wandb"] = False
        experiment_config["plugins"]["wandb"] = None
    if args["dryrun"]:
        WANDB_MODE_bck = os.environ.get("WANDB_MODE", None)
        os.environ["WANDB_MODE"] = 'dryrun'
    run_dir = setup_experiment(experiment_config)
    
    # Verbosity
    quiet = args["quiet"]

    # Run trainong process
    model = train_model(experiment_config, quiet=quiet, run_dir=run_dir, display=GUIcmd())

    # restore configuration
    if args["dryrun"]:
        os.environ["WANDB_MODE"] = WANDB_MODE_bck