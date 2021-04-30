import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *
from pathlib import Path


def train_aleatoric(config, quiet=False, run_dir="."):

    # Experiment paths and indentifiers
    experiments_folder = config["experiment"]["experiments_folder"]
    experiment_name    = config["experiment"]["name"]
    experiment_folder  = Path(experiments_folder) / experiment_name
    run_id             = config["experiment"]["run_id"]
    
    # Training parameters
    epochs = config["training"]["epochs"]
    batch_size  = config["training"]["batch_size"]    

    # Dataset
    data_configuration = config["data"]
    generators, input_shape, num_classes, labels = setup_generators(
        batch_size=batch_size, **data_configuration
    )
    train_, val_ = generators
    (train_dataset, steps_per_epoch) = train_
    (val_dataset, validation_steps)  = val_

    # Model architecture
    model_configuration = config["model"]
    model = setup_model(input_shape, num_classes, **model_configuration, build=True)
    aleatoric_model = model.get_aleatoric_model()

    ## Loss
    loss = aleatoric_model.build_aleatoric_loss()

    ## Optimizer
    optimizer_name = config["training"]["optimizer"]["name"]
    optimizer_learning_rate = float(config["training"]["optimizer"]["learning_rate"]) # fix scientific notation parsed as str.
    optimizer_parameters = config["training"]["optimizer"].get("parameters", {})
    optimizer_parameters = optimizer_parameters or {}
    optimizer = OPTIMIZERS[optimizer_name](
        learning_rate=optimizer_learning_rate, 
        **optimizer_parameters
    )
    ## Class Weights
    class_weight = None
    class_weight_list = config["training"].get("class_weight", None)
    if class_weight_list is not None:
        class_weight = {i: w for i, w in enumerate(class_weight_list)}
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
         run_dir=run_dir,
         uncertainty_type="aleatoric"
    )
    
    # Train
    aleatoric_model.compile(
        optimizer=optimizer,
        loss=loss
    )
    history = aleatoric_model.fit(train_dataset, 
        verbose = 2 if quiet else 1,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset, 
        validation_steps=validation_steps,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks
    )
    
    # Return final model
    return aleatoric_model

if __name__ == "__main__":
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_DISABLE_PTX_JIT'] = "1"
    import argparse
    ap = argparse.ArgumentParser(description="Train aleatoric version of model.")
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
    ap.add_argument("--seed", type=int, default=None, 
        help="Overwrite experiment`s seed.")
    args = vars(ap.parse_args()) 

    # Load experiment configuration
    config_file = args["config"]
    print(f"Loading config from: {config_file}")
    experiment_config = load_config_file(config_file)

    # Configure multiples runs
    job = args["job"]
    if job is not None:
        experiment_config["experiment"]["tags"].append(f"job {job}")
        if experiment_config["experiment"]["seed"] is not None:
            experiment_config["experiment"]["seed"] += job
    # Overwrite seed
    seed = args["seed"]
    if seed is not None:
        experiment_config["experiment"]["seed"]= seed
    # Setup experiments and pluggins
    if args["disable_wandb"]:
        experiment_config["training"]["callbacks"]["enable_wandb"] = False
        experiment_config["plugins"]["wandb"] = None
    if args["dryrun"]:
        WANDB_MODE_bck = os.environ.get("WANDB_MODE", None)
        os.environ["WANDB_MODE"] = 'dryrun'
    # Add Aleatoric modifications
    if "aleatoric" not in experiment_config["experiment"]["tags"]:
        experiment_config["experiment"]["tags"].append("aleatoric")

    run_dir = setup_experiment(experiment_config, mode="training")
    
    # Verbosity
    quiet = args["quiet"]

    # Run training process
    model = train_aleatoric(experiment_config, quiet=quiet, run_dir=run_dir)

    # restore configuration
    if args["dryrun"]:
        os.environ["WANDB_MODE"] = WANDB_MODE_bck