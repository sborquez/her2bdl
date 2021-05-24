import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *
from pathlib import Path
from .train_model import train_model

def train_aleatoric(config, quiet=False, run_dir="."):
    # Return aleatoric model
    return train_model(config, 
                       uncertainty_type="aleatoric", quiet=quiet, run_dir=run_dir)

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
        WANDB_MODE_bck = os.environ.get("WANDB_MODE", "")
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