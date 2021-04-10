import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from her2bdl import *

from pathlib import Path
import logging
import time
from tensorflow.random import set_seed
import numpy as np


def evaluate(config, quiet=False, run_dir="."):
    # Experiment paths and indentifiers
    experiments_folder = config["experiment"]["experiments_folder"]
    experiment_name    = config["experiment"]["name"]
    experiment_folder  = Path(experiments_folder) / experiment_name
    run_id             = config["experiment"]["run_id"]
    
    # Evaluation parameters
    batch_size  = config["evaluate"]["batch_size"]

    # Dataset
    data_configuration = config["data"]
    test_, input_shape, num_classes, labels = setup_generators(
        #batch_size=batch_size, #TODO: add eval parameters
        test_dataset=True,  **data_configuration
    )
    (test_dataset, steps_per_epoch) = test_

    # Model architecture
    model_configuration = config["model"]
    model = setup_model(input_shape, num_classes, **model_configuration)
    # Aggregator method
    aggregator_configuration = config["aggregation"]
    aggregator = setup_aggregator(**aggregator_configuration)

    # Patch Evaluation
    enable_wandb  = config["evaluate"]["experiment_logger"]["enable_wandb"]
    patch_metrics = config["evaluate"]["experiment_logger"]["patch_metrics"]
    wsi_metrics = config["evaluate"]["experiment_logger"]["wsi_metrics"]
    uncertainty_metric = config["evaluate"]["experiment_logger"]["uncertainty_metric"]
    logger = setup_evaluation_logger(
        model_name=experiment_name,
        labels=labels,
        enable_wandb=enable_wandb,
        patch_metrics=patch_metrics,
        wsi_metrics=wsi_metrics,
        uncertainty_metric=uncertainty_metric,
        run_dir=run_dir

    )
    ## Get performance metrics for patch classification.
    results = model.predict_with_epistemic_uncertainty(test_dataset)
    predictions_results, uncertainty_results = results
    logger.log_patch_classification(
        predictions_results, uncertainty_results
    )

    # Whole Image Evaluation
    ## Use aggregation and get performance metrics at wsi level.
    tissue_results={}
    tissue_y_true = []
    tissue_y_pred = []
    tissue_y_predictive_distribution = []
    for group_partition in test_dataset.get_partition(predictions_results, uncertainty_results, by="CaseNo_label"):
        group, group_df, group_predictions_results, group_uncertainty_results = group_partition
        prediction_agg_result, uncertainty_agg_result = aggregator(group_predictions_results, group_uncertainty_results)
        tissue_results[group] = {
            **prediction_agg_result, 
            **uncertainty_agg_result,
        }
        tissue_y_true.append(group_df.iloc[0][TARGET])
        tissue_y_pred.append(prediction_agg_result['y_pred'])
        tissue_y_predictive_distribution.append(prediction_agg_result['y_predictive_distribution'])
    tissue_y_true = np.array(tissue_y_true)
    tissue_y_pred = np.array(tissue_y_pred)
    tissue_y_predictive_distribution = np.array(tissue_y_predictive_distribution)
    ## Get performance metrics for tissue classification.

    # Whole Image Uncertainty Evaluation

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate models.")
    ap.add_argument("-c", "--config", type=str, required=False, default=None, 
        help="Configuration file for evaluation.")
    ap.add_argument("-e", "--experiment", type=str, required=False, default=None, 
        help="Experiment run folder for evaluation.")
    ap.add_argument("--dryrun", action='store_true', 
        help="Run locally. Upload your results to wandb servers afterwards.")
    ap.add_argument("--quiet", action='store_true', 
        help="Disable progress bar.") 
    ap.add_argument("--disable_wandb", action='store_true', 
        help="Disable WandB for locally testing without Weight&Bias Callbacks.")
    ap.add_argument("--job", type=int, default=None, 
        help="Disable WandB for locally testing without Weight&Bias Callbacks.")
    ap.add_argument("-o", "--output", type=str,
        default="D:/sebas/Projects/her2bdl/train/experiments/runs",
        help="Change experiment runs output folder.")
    args = vars(ap.parse_args())
    # Overwrite configure
    overwrite_config = {}
    output_folder = args["output"]
    if output_folder is not None: 
        overwrite_config["experiment"] = {
            "experiments_folder": output_folder
        }
    # Load experiment configuration
    config_file = args["config"]
    experiment_folder = args["experiment"]
    if experiment_folder is not None:
        print(f"Loading experiment from: {experiment_folder}")
        # TODO: implement this configuration in a more flexible way.
        # maybe by using the .env file
        overwrite_config["plugins"] =  {
            "wandb": {
                "project": "Her2BDL",
                "apikey": "WANDB_API_KEY"
            }
        }
        experiment_config = load_run_config(experiment_folder, **overwrite_config)
    elif config_file is not None:
        print(f"Loading config from: {config_file}")
        experiment_config = load_config_file(config_file, **overwrite_config)
    else:
        raise ValueError("Required configuration file or experiment folder.")
    # Configure multiples runs
    job = args["job"]
    if job is not None:
        model_name = experiment_config["experiment"]["name"]
        job_sufix  = f"job {str(job).zfill(2)}"
        experiment_config["experiment"]["name"]= f"{model_name} {job_sufix}"
    # Setup experiments and plugins
    if args["disable_wandb"]:
        experiment_config["evaluation"]["experiment_logger"]["enable_wandb"] = False
        experiment_config["plugins"]["wandb"] = None
    if args["dryrun"]:
        WANDB_MODE_bck = os.environ.get("WANDB_MODE", None)
        os.environ["WANDB_MODE"] = 'dryrun'
    run_dir = setup_experiment(experiment_config, mode="evaluation")

    # Verbosity
    quiet = args["quiet"]

    # Run evaluation process
    results = evaluate(experiment_config, quiet=quiet, run_dir=run_dir)

    # restore configuration
    if args["dryrun"]:
        os.environ["WANDB_MODE"] = WANDB_MODE_bck