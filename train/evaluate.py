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
    batch_size  = config["evaluation"]["batch_size"]
    evaluate_classification = config["evaluation"]["evaluate_classification"]
    evaluate_aleatoric_uncertainty = config["evaluation"]["evaluate_aleatoric_uncertainty"]
    evaluate_aggregation = config["evaluation"]["evaluate_aggregation"]
    # Dataset
    data_configuration = config["data"]
    test_, input_shape, num_classes, labels = setup_generators(
        batch_size=batch_size, test_dataset=True,  **data_configuration
    )
    (test_dataset, steps_per_epoch) = test_
    # Model architecture
    model_configuration = config["model"]
    model = setup_model(input_shape, num_classes, **model_configuration)
    # Evaluation Logguer
    logger_configuration = config["evaluation"]["experiment_logger"]
    logger = setup_evaluation_logger(
        model_name=experiment_name,
        labels=labels,
        run_dir=run_dir,
        **logger_configuration
    )
    # Evaluation options
    if evaluate_classification:
        ## Get performance metrics for classification and epistemic uncertainty.
        results = model.predict_with_epistemic_uncertainty(test_dataset, include_data=True)
        data, predictions_results, uncertainty_results = results
        logger.log_classification_metrics(data, predictions_results, uncertainty_results)
    if evaluate_aleatoric_uncertainty:        
        ## Get metrics for Aleatoric uncertainty.
        aleatoric_model = model.get_aleatoric_model()
        results_aleatoric = aleatoric_model.predict_with_aleatoric_uncertainty(test_dataset, include_data=True)
        aleatoric_data, aleatoric_predictions_results, aleatoric_uncertainty_results = results_aleatoric
        logger.log_aleatoric_uncertainty(aleatoric_data, aleatoric_predictions_results, aleatoric_uncertainty_results)
    if evaluate_aggregation:
        # Aggregator method
        aggregator_configuration = config["aggregation"]
        aggregator = setup_aggregator(**aggregator_configuration)
        if results is None:
            results = model.predict_with_epistemic_uncertainty(test_dataset, include_data=True)
        data, predictions_results, uncertainty_results = results
        agg = aggregator.predict_with_aggregation(test_dataset, predictions_results, uncertainty_results, include_data=True, verbose=1)
        aggregated_data, aggregated_predictions, aggregated_uncertainty = agg
        logger.log_aggregation_metrics(
            test_dataset,
            data, predictions_results, uncertainty_results,
            aggregated_data, aggregated_predictions, aggregated_uncertainty,
        )
    return


if __name__ == "__main__":
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_DISABLE_PTX_JIT'] = "1"
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate models.")
    ap.add_argument("-c", "--config", type=str, required=True, 
        help="Configuration file for evaluation.")
    #ap.add_argument("-e", "--experiment", type=str, required=False, default=None, 
    #    help="Experiment run folder for evaluation.")
    ap.add_argument("--dryrun", action='store_true', 
        help="Run locally. Upload your results to wandb servers afterwards.")
    ap.add_argument("--quiet", action='store_true', 
        help="Disable progress bar.")
    ap.add_argument("--job", type=int, default=None, 
        help="Disable WandB for locally testing without Weight&Bias Callbacks.")
    ap.add_argument("-o", "--output", type=str, default=None,
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
    if config_file is not None:
        print(f"Loading config from: {config_file}")
        experiment_config = load_config_file(config_file, **overwrite_config)
    # TODO: implement this configuration in a more flexible way.
    # experiment_folder = args["experiment"]
    # if experiment_folder is not None:
    #     print(f"Loading experiment from: {experiment_folder}")
    #     # maybe by using the .env file
    #     overwrite_config["plugins"] =  {
    #         "wandb": {
    #             "project": "Her2BDL",
    #             "apikey": "WANDB_API_KEY"
    #         }
    #     }
    #     experiment_config = load_run_config(experiment_folder, **overwrite_config)
    else:
        raise ValueError("Required configuration file or experiment folder.")
    # Configure multiples runs
    job = args["job"]
    if job is not None:
        model_name = experiment_config["experiment"]["name"]
        job_sufix  = f"job {str(job).zfill(2)}"
        experiment_config["experiment"]["name"]= f"{model_name} {job_sufix}"
    # Setup experiments and plugins
    if args["dryrun"]:
        WANDB_MODE_bck = os.environ.get("WANDB_MODE", "")
        os.environ["WANDB_MODE"] = 'dryrun'
    run_dir = setup_experiment(experiment_config, mode="evaluation")

    # Verbosity
    quiet = args["quiet"]

    # Run evaluation process
    results = evaluate(experiment_config, quiet=quiet, run_dir=run_dir)

    # restore configuration
    if args["dryrun"]:
        os.environ["WANDB_MODE"] = WANDB_MODE_bck