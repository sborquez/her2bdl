"""
Utility tools for train/evaluate experiments
================================================

Collection of functions shared between train scripts.
"""

import os
from os.path import join
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow.random import set_seed

import yaml
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from .models.metrics import (
    confusion_matrix, class_stat, overall_stat, multiclass_roc_curve
)
from .visualization.performance import (
    display_confusion_matrix, display_multiclass_roc_curve
)
from .visualization.prediction import (
    display_prediction, display_uncertainty, display_uncertainty_by_class
)
from .models import MODELS, AGGREGATION_METHODS
from .data import get_generator_from_wsi, get_generators_from_tf_Dataset
from .data import TARGET_LABELS_list


__all__ = [
    "load_config_file", "load_run_config",
    "setup_experiment", "setup_model", "setup_aggregator",
    "setup_generators", "setup_callbacks", "setup_evaluation_logger"
]

# Experiment files variables
__sections = set([
    "experiment", "model", "aggregation", "data",
    "training", "evaluation", "predict", "plugins",
])

__required = {
    "experiment":
        ["name", "notes", "experiments_folder"],
    "model":
        ["task", "architecture", "hyperparameters", "uncertainty"],
    "aggregation":
        ["method", "parameters"],
    "data":
        ["source",
        "img_height", "img_width", "img_channels",
        "num_classes"
        ],
    "training":
        ["epochs", "loss", "callbacks"],
    "evaluation":
        #["metrics"],
        [],
    "predict":
        ["save_aggregation", "save_predictions", "save_uncertainty"]
}

__default_optional = {
    "experiment": {"tags": [], "run_id": None, "seed": 1234},
    "model": {"weights": None, "aleatoric_weights": None},
    "aggregation": {},
    "data": {"preprocessing": {}, "label_mode": "categorical", "labels": None},
    "training": {
        "batch_size": 16, "validation_split": 0.2,
        "optimizer": {
            "name": "adam", "learning_rate": 0.4, "parameters": None
        },
    },
    "evaluation": {
        "batch_size": 16,
        "evaluate_classification": True,
        "evaluate_aleatoric_uncertainty": True,
        "evaluate_aggregation": False,
        "experiment_logger": {
                "enable_wandb" : True, "classification_metrics": {}, 
                "aggregation_metrics": {}, "aleatoric_uncertainty": {},
                "max_plots": 120
            }
    },
    "predict": {},
    "plugins":{"wandb": None}
}


def parse_wandb_config(config):
    parsed_config = {}
    for section in __sections:
        if section in (set(config.keys()) - set(("wandb_version", "_wandb"))):
            parsed_config[section] = config.get(section)["value"]
    return parsed_config


def load_config_file(config_filepath, **overwrite_config):
    """Load, validate and set default values"""
    # Read config file
    with open(config_filepath) as yaml_file:
        base_config = yaml.safe_load(yaml_file)
        if "wandb_version" in base_config:
            base_config = parse_wandb_config(base_config)
    # Check sections
    assert set(base_config.keys()).issubset(__sections),\
     "Configuration file has missings sections."
    # Check required sections and subsections
    not_defined = {}
    for section, required_keys in __required.items():
        defined_keys = base_config[section].keys()
        if len(set(required_keys) - set(defined_keys)) > 0:
            not_defined[section] = set(required_keys) - set(defined_keys)
    assert len(not_defined) == 0, f"Configuration not defined: {not_defined}"
    # Default values
    config = __default_optional.copy()
    
    for section in base_config.keys():
        config[section].update(base_config[section])
    for section in overwrite_config:
        config[section].update(overwrite_config[section])
    
    # Set Experiment run id
    if config["experiment"]["run_id"] is None:
        config["experiment"]["run_id"] = wandb.util.generate_id()
    return config


def load_run_config(run_folderpath, **overwrite_config):
    run_folder = Path(run_folderpath)
    # is wandb run
    if len(list(run_folder.glob('run-*.wandb'))) > 0:
        config_filepath = run_folder / "files" / "config.yaml"
        best_model = run_folder / "files" / "model-best.h5"
        if "experiment" not in overwrite_config:
            overwrite_config["experiment"] = dict()
        overwrite_config["experiment"]["run_id"] = None
    # is local
    else:
        config_filepath = run_folder / "config.yaml"
        best_model = sorted((run_folder / "checkpoints").glob("*.h5"))[-1]
    config = load_config_file(config_filepath, **overwrite_config)
    config["experiment"]["run_folder"] = run_folder.as_posix()
    config["model"]["weights"] = best_model
    return config


def setup_experiment(experiment_config, mode="training"):
    # Seed
    seed = experiment_config["experiment"]["seed"]
    if seed is not None:
        np.random.seed(seed)
        set_seed(seed)
    # Weight and Bias
    plugins = experiment_config.get("plugins", {})
    del experiment_config["plugins"]
    wand_config = plugins.get("wandb", None)
    if wand_config is not None:
        assert wand_config["apikey"] is not None,\
            "Requiere enviroment variable 'WANDB_API_KEY' or API Key."
        if wand_config["apikey"]  == 'WANDB_API_KEY':
            assert wand_config["apikey"] in os.environ,\
                "Requiere enviroment variable 'WANDB_API_KEY' defined."
        elif Path(wand_config["apikey"]).is_file() and \
                Path(wand_config["apikey"]).stem == '.wandb_secret':
            with open(wand_config["apikey"]) as secret:
                os.environ["WANDB_API_KEY"] = secret.read().strip()
        else:
            os.environ["WANDB_API_KEY"] = wand_config["apikey"]
        project = wand_config["project"]
        wandb.init(
            project = project,
            name    = experiment_config["experiment"]["name"],
            notes   = experiment_config["experiment"]["notes"],
            tags    = experiment_config["experiment"]["tags"],
            dir     = experiment_config["experiment"]["experiments_folder"],
            job_type= mode,
            id      = experiment_config["experiment"]["run_id"],
            config  = experiment_config
        )
        return wandb.run.dir
    else:
        if mode == "training":
            # for consistency
            experiment_config["training"]["callbacks"]["enable_wandb"] = False 
            pass
        elif mode == "evaluate":
            # TODO: add enable_wandb option for evaluation
            experiment_config["evaluate"]["experiment_logger"]["enable_wandb"] = False 
            pass
        else:
            raise ValueError(f"mode {mode} not supported yet.")
        experiments_folder = \
            experiment_config['experiment']['experiments_folder']
        run_id = experiment_config["experiment"]["run_id"]
        folder_ = f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{run_id}"
        experiment_folder = join(experiments_folder, folder_)
        os.makedirs(experiment_folder, exist_ok=True)
        with open(join(experiment_folder, 'config.yaml'), 'w') as outfile:
            yaml.dump(experiment_config, outfile)
        return experiment_folder


def setup_model(input_shape, num_classes, architecture, uncertainty,
                hyperparameters, weights=None, aleatoric_weights=None,
                task=None, build=False, build_aleatoric=False):
    """
    Construct model from configuration.
    Parameters
    ----------
    input_shape : `tuple` (img_widht, img_height, img_channels)
        Model's input shape.
    num_classes: `ìnt`
        Output shape, number of classes.
    architecture : `str`
        Model architecture defined in models submodule.
    hyperparameters :  `dict`
        Model's hyperparameters defined in configuration file.
    uncertainty :  `dict`
        model's uncertainty hyperparameters defined in configuration file.
    weights : `str`
        Path to model`s trained weights or checkpoint.
    build :  `bool`
        Use `build=True` for inspect the model without compiling it.
    build_aleatoric : `bool`
        Use `build_aleatoric=True` for inspect the aleatoric model without compiling it.
    task : (ignored)
    Return
    ------
        `her2bdl.models.mcdropout.MCDropoutModel`.
    """
    if architecture not in MODELS: 
        raise ValueError(f"Unknown architecture: {architecture}")
    base_model = MODELS[architecture]
    model = base_model(
        input_shape, num_classes, 
        **hyperparameters, #config["model"]["hyperparameters"], 
        **uncertainty,     #config["model"]["uncertainty"]
    )
    if build or (weights is not None):
        model(np.empty((1, *input_shape), np.float32))
        if weights is not None:
            model.load_weights(weights)
    if build_aleatoric or (aleatoric_weights is not None):
        aleatoric_model = model.get_aleatoric_model()
        aleatoric_model(np.empty((1, *input_shape), np.float32))
        if aleatoric_weights is not None:
            model.load_aleatoric_weights(aleatoric_weights)
    return model

def setup_aggregator(method, parameters):
    """
    Construct aggregator from configuration.
    
    Parameters
    ----------
    method : `str`
        Aggregation method in models submodule.
    parameters :  `dict`
        Aggregator's parameters defined in configuration file.
    Return
    ------
        `her2bdl.models.aggregation.Aggregator`.
    """
    if method not in AGGREGATION_METHODS: 
        raise ValueError(f"Unknown aggregation method: {method}")
    base_aggregator = AGGREGATION_METHODS[method]
    aggregator = base_aggregator(
        **parameters
    )
    return aggregator

def setup_generators(source, num_classes, labels, label_mode, preprocessing, 
                     img_height=240, img_width=240, img_channels=3, 
                     validation_split=None, batch_size=16, test_dataset=False):
    # Parameters
    input_shape = (img_height, img_width, img_channels)
    labels = labels
    if labels == "HER2": labels = TARGET_LABELS_list
    source_type = source["type"]
    dataset_parameters = source["parameters"]
    # Load train and validation generators
    if source_type == "tf_Dataset":
        if not test_dataset:
            generators = get_generators_from_tf_Dataset(
                **dataset_parameters, 
                num_classes=num_classes, label_mode=label_mode,
                input_shape=input_shape, batch_size=batch_size, 
                validation_split=validation_split, preprocessing=preprocessing
            )
        else:
            raise NotImplementedError
    elif source_type == "wsi":
        if not test_dataset:
            #Load train and validation generators 
            generators = get_generator_from_wsi(
                generator =  dataset_parameters["train_generator"],
                validation_generator = dataset_parameters.get("validation_generator", None),
                num_classes=num_classes, label_mode=label_mode,
                input_shape=input_shape, batch_size=batch_size,
                preprocessing=preprocessing
            )
        else:
            # Load test generator
            generators = get_generator_from_wsi(
                generator = dataset_parameters["test_generator"],
                num_classes = num_classes, label_mode = label_mode,
                input_shape = input_shape, batch_size = batch_size,
                preprocessing = preprocessing
            )
    else:
        raise ValueError(f"Unknown source_type: {source_type}")
    return generators, input_shape, num_classes, labels

def setup_callbacks(validation_data, validation_steps, model_name, batch_size, 
    	           enable_wandb=True, labels=None, earlystop=None, uncertainty_type="epistemic",
                   experiment_tracker=None, checkpoints=None, run_dir="."):
    """
    Callbacks configuration:
        # Early stop, use null to disable.
        earlystop:
            patience: 15
            monitor: val_loss
        # Experiments results while training
        experiment_tracker:
            # Save model architecture summary
            (ignored) log_model_summary: true
            # Save datasets info: source, sizes, etc.
            (ignored) log_dataset_description: true
            # Plots and logs
            (ignored) log_training_loss_plot: true 
            log_predictions_plot: false
            log_predictions: false
            log_uncertainty_plot: false
            log_uncertainty: false
            log_confusion_matrix_plot: false
        # Save checkpoints: saved at experiments_folder/experiment/checkpoints
        checkpoints:
            # saved weights format
            format: "weights.{epoch:03d}-{val_loss:.4f}.h5"
            save_best_only: true
            save_weights_only: true
            monitor: val_loss
    """
    callbacks =[]
    if enable_wandb:
        if uncertainty_type == "epistemic":
            uncertainty_callback = EpistemicCallback
        else:
            uncertainty_callback = AleatoricCallback
        callbacks.append(
            uncertainty_callback(
                generator=validation_data,
                validation_steps=len(validation_data),
                model_name=model_name,
                batch_size=batch_size,
                labels=labels,
                **experiment_tracker
            )
        )
    if checkpoints is not None:
        os.makedirs(join(run_dir, "checkpoints"), exist_ok=True)
        filepath_format = checkpoints["format"]
        del checkpoints["format"]
        filepath = join(run_dir, "checkpoints", filepath_format)
        callbacks.append(ModelCheckpoint(filepath, **checkpoints))
    if earlystop is not None:
        callbacks.append(EarlyStopping(**earlystop))
    return callbacks


"""
Callbacks and Evaluation metrics loggers
----------------------------------------
"""


class WandBLogGenerator():
    """
    WandB plots and metrics formater.
    """

    def __init__(self, model_name, labels=None, max_plots=8):
        self.model_name = model_name
        self.labels     = labels
        self.max_plots  = max_plots

    def _log_metrics(self, y_pred, y_true):
        """
        Get confussion matrix metrics, and formated into tables.
        
        Parameters
        ----------
        y_pred : `np.ndarray`
            Predicted classes. (batch, 1)
        y_true : `np.ndarray`
            True classes. (batch, 1)
        Return
        ------
            `(wandb.Table, wandb.Table, dict)`
                Return class and overall metrics, on table and dict format.
        """
        labels = self.labels
        # Individual class metrics
        class_stat_df = class_stat(y_true, y_pred, labels=labels)
        class_stats_table = wandb.Table(dataframe=class_stat_df)
        class_stats_dict  = {}
        for i, row in class_stat_df.iterrows():
            class_stat_prefix = row["class stat"]
            for metric, value in row.items():
                if metric == "class stat": continue
                class_stats_dict[f"{class_stat_prefix} - Class {metric}"] = value
        # Overall metrics: micro and macro averages.
        overall_stat_df = overall_stat(y_true, y_pred, labels=labels)
        overall_stats_table = wandb.Table(dataframe=overall_stat_df)
        overall_stat_dict = {
            row["overall stat"]:row["value"] 
            for i, row in overall_stat_df.iterrows()
        }
        # 
        overall_and_class_values = {**overall_stat_dict, **class_stats_dict}
        return class_stats_table, overall_stats_table, overall_and_class_values
    
    def _log_confusion_matrix(self, y_pred, y_true):
        """
        Get confussion matrix plot.
        
        Parameters
        ----------
        y_pred : `np.ndarray`
            Predicted classes. (batch, 1)
        y_true : `np.ndarray`
            True classes. (batch, 1)
        Return
        ------
            `wandb.Image`
                Confussion matrix plot.
        """
        labels = self.labels
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        figure = display_confusion_matrix(
            cm=cm,
            model_name=self.model_name,
            labels=labels
        )
        img_log = wandb.Image(figure)
        plt.close(figure)
        return img_log

    def _log_roc_curve(self, y_prob, y_true):
        """
        Get ROC plot.
        
        Parameters
        ----------
        y_prob : `np.ndarray`
            Predicted probability classes. (batch, classes)
        y_true : `np.ndarray`
            True classes. (batch, 1)
        Return
        ------
            `wandb.Image`
                ROC curve plot.
        """
        fpr, tpr, roc_auc = multiclass_roc_curve(y_true, y_prob)
        figure = display_multiclass_roc_curve(
            fpr, tpr, roc_auc,
            model_name=self.model_name,
            labels=self.labels
        )
        img_log = wandb.Image(figure)
        plt.close(figure)
        return img_log

    def _log_predictions(self, x_true, y_true, 
                         y_predictive_distribution, y_pred):
        """
        Display sample and models prediction.
        
        Parameters
        ----------
        x_true : `np.ndarray`
            True classes. (batch, 1)
        y_true : `np.ndarray`
            True classes. (batch, 1)
        y_predictive_distribution : `np.ndarray`
            Predicted distribution. (batch, ?, ?)
        y_pred : `np.ndarray`
            Predicted classes. (batch, 1)
        Return
        ------
            List of `wandb.Image`
                Prediction and input image samples.
        """
        wandb_images = []
        length = len(x_true)
        if length > self.max_plots :
            choices = np.random.choice(
                range(length), size=self.max_plots , replace=False
            )
        else:
            choices = range(length)
        for i in choices:
            figure = display_prediction(
                x = x_true[i],
                y_pred = y_pred[i],
                y_true = y_true[i], 
                predictive_distribution = y_predictive_distribution[i],
                model_name = self.model_name,
                labels = self.labels
            )
            wandb_images.append(
                wandb.Image(figure)
            )
            plt.close(figure)
        return wandb_images

    def _log_mean_uncertainty(self, uncertainty):
        """
        Get mean uncertainty values.
        
        Parameters
        ----------
        uncertainty : `pd.DataFrame`
        Return
        ------
            `dict`
                Pairs uncertainty metric and its mean value.
        """
        mean = dict(uncertainty.mean())
        return {f"mean_{k}":v for k,v in mean.items()}

    def _log_epistemic_uncertainty(self, x_true, y_true, 
                         y_predictive_distribution, y_pred, 
                         y_predictions_samples, uncertainty):
        """
        Display sample and models uncertainty.

        Parameters
        ----------
        x_true : `np.ndarray`
            True classes. (batch, 1)
        y_true : `np.ndarray`
            True classes. (batch, 1)
        y_predictive_distribution : `np.ndarray`
            Predicted distribution. (batch, ?, ?)
        y_pred : `np.ndarray`
            Predicted classes. (batch, 1)
        y_predictions_samples : `np.ndarray`
            Predicted classes samples. (batch, samples, classes)
        uncertainty : `pd.DataFrame`
        Return
        ------
            List of `wandb.Image`
                Prediction, input image and uncertainty samples.
        """
        wandb_images = []
        length = len(x_true)
        if length > self.max_plots :
            choices = np.random.choice(range(length), 
                        size=self.max_plots , replace=False)
        else:
            choices = range(length)
        for i in choices:
            figure = display_uncertainty(
                x = x_true[i],
                y_pred = y_pred[i],
                y_true = y_true[i], 
                predictive_distribution = y_predictive_distribution[i],
                prediction_samples = y_predictions_samples[i],
                uncertainty = uncertainty.iloc[i],
                model_name = self.model_name,
                labels = self.labels
            )
            wandb_images.append(
                wandb.Image(figure)
            )
            plt.close(figure)
        return wandb_images

    def _log_uncertainty_by_class(self, y_true, uncertainty):
        """
        Display uncertainty boxplots.

        Parameters
        ----------
        y_true : `np.ndarray`
            True classes. (batch, 1)
        uncertainty : `pd.DataFrame`

        Return
        ------
            List of `wandb.Image`
                uncertainty boxplots.
        """
        wandb_images = []
        for metric in uncertainty.columns:
            figure = display_uncertainty_by_class(
                y_true, uncertainty, metric, 
                model_name=self.model_name,
                labels = self.labels
            )
            wandb_images.append(
                wandb.Image(figure, caption=f"{metric} by class")
            )
            plt.close(figure)
        return wandb_images

    def _log_epistemic_uncertainty_highlights(self, x_true, y_true, 
                                    y_predictive_distribution, y_pred, 
                                    y_predictions_samples, uncertainty, 
                                    mode='max'):
        """
        Display sample uncertainty with higher or lower uncertainty.

        Parameters
        ----------
        x_true : `np.ndarray`
            True classes. (batch, 1)
        y_true : `np.ndarray`
            True classes. (batch, 1)
        y_predictive_distribution : `np.ndarray`
            Predicted distribution. (batch, ?, ?)
        y_pred : `np.ndarray`
            Predicted classes. (batch, 1)
        y_predictions_samples : `np.ndarray`
            Predicted classes samples. (batch, samples, classes)
        uncertainty : `pd.DataFrame`
        mode :  `str` ['max' | 'min'] (default='max')
            Select mode for sorted samples.
        Return
        ------
            List of `wandb.Image`
                Prediction, input image and uncertainty samples.
        """
        # TODO: Merge this with the epistemic highlight, to dont repeat code
        caption = "higher" if mode == "max" else "lower"
        highlights = []
        for metric in uncertainty:
            if mode == 'max':
                index = uncertainty[metric].argmax()
            elif mode == 'min':
                index = uncertainty[metric].argmin()
            highlights.append((index, metric))
        wandb_images = []
        for i, metric in highlights:
            figure = display_uncertainty(
                x = x_true[i],
                y_pred = y_pred[i],
                y_true = y_true[i], 
                predictive_distribution = y_predictive_distribution[i],
                prediction_samples = y_predictions_samples[i],
                uncertainty = uncertainty.iloc[i],
                model_name = self.model_name,
                labels = self.labels
            )
            wandb_images.append(
                wandb.Image(figure, caption=f"{caption} {metric}")
            )
            plt.close(figure)
        return wandb_images

    def _log_aleatoric_uncertainty(self, x_true, y_true, 
                         y_predictive_distribution, y_pred, uncertainty):
        """
        Display sample and models uncertainty.

        Parameters
        ----------
        x_true : `np.ndarray`
            True classes. (batch, 1)
        y_true : `np.ndarray`
            True classes. (batch, 1)
        y_predictive_distribution : `np.ndarray`
            Predicted distribution. (batch, ?, ?)
        y_pred : `np.ndarray`
            Predicted classes. (batch, 1)
        uncertainty : `pd.DataFrame`
        Return
        ------
            List of `wandb.Image`
                Prediction, input image and uncertainty samples.
        """
        wandb_images = []
        length = len(x_true)
        if length > self.max_plots :
            choices = np.random.choice(range(length), 
                        size=self.max_plots , replace=False)
        else:
            choices = range(length)
        for i in choices:
            figure = display_uncertainty(
                x = x_true[i],
                y_pred = y_pred[i],
                y_true = y_true[i], 
                predictive_distribution = y_predictive_distribution[i],
                prediction_samples = None,
                uncertainty = uncertainty.iloc[i],
                model_name = self.model_name,
                labels = self.labels
            )
            wandb_images.append(
                wandb.Image(figure)
            )
            plt.close(figure)
        return wandb_images

    def _log_aleatoric_uncertainty_highlights(self, x_true, y_true, 
                                    y_predictive_distribution, y_pred, uncertainty, 
                                    mode='max'):
        """
        Display sample uncertainty with higher or lower uncertainty.

        Parameters
        ----------
        x_true : `np.ndarray`
            True classes. (batch, 1)
        y_true : `np.ndarray`
            True classes. (batch, 1)
        y_predictive_distribution : `np.ndarray`
            Predicted distribution. (batch, ?, ?)
        y_pred : `np.ndarray`
            Predicted classes. (batch, 1)
        uncertainty : `pd.DataFrame`
        mode :  `str` ['max' | 'min'] (default='max')
            Select mode for sorted samples.
        Return
        ------
            List of `wandb.Image`
                Prediction, input image and uncertainty samples.
        """
        # TODO: Merge this with the epistemic highlight, to dont repeat code
        caption = "higher" if mode == "max" else "lower"
        highlights = []
        for metric in uncertainty:
            if mode == 'max':
                index = uncertainty[metric].argmax()
            elif mode == 'min':
                index = uncertainty[metric].argmin()
            highlights.append((index, metric))
        wandb_images = []
        for i, metric in highlights:
            figure = display_uncertainty(
                x = x_true[i],
                y_pred = y_pred[i],
                y_true = y_true[i], 
                predictive_distribution = y_predictive_distribution[i],
                prediction_samples = None,
                uncertainty = uncertainty.iloc[i],
                model_name = self.model_name,
                labels = self.labels
            )
            wandb_images.append(
                wandb.Image(figure, caption=f"{caption} {metric}")
            )
            plt.close(figure)
        return wandb_images

class EpistemicCallback(wandb.keras.WandbCallback, WandBLogGenerator):
    """
    WandbCallback extension for include extra plots and summaries for a epistemic model (MCModel).
    """

    def __init__(self, generator, validation_steps, model_name, batch_size,
                 monitor='val_loss', verbose=0, mode='auto',
                 save_weights_only=True, log_weights=True, log_gradients=False,
                 save_model=True, training_data=None, validation_data=None,
                 labels=None, data_type=None, predictions=200, input_type="image",
                 output_type="label", log_evaluation=False, class_colors=None,
                 log_batch_frequency=None, log_best_prefix="best_",
                 log_confusion_matrix=True, log_predictions=True, 
                 log_uncertainty=True, log_metrics=True, log_roc_curve=True, **kwawgs):
        WandBLogGenerator.__init__(self,
            model_name=model_name,
            labels=labels
        )
        wandb.keras.WandbCallback.__init__(self,
                        monitor=monitor,
                        verbose=verbose,
                        mode=mode,
                        save_weights_only=save_weights_only,
                        log_weights=log_weights,
                        log_gradients=log_gradients,
                        save_model=save_model,
                        training_data=training_data,
                        validation_data=validation_data,
                        labels=labels,
                        data_type=data_type,
                        predictions=predictions,
                        generator=generator,
                        input_type=input_type,
                        output_type=output_type,
                        log_evaluation=log_evaluation,
                        validation_steps=validation_steps,
                        class_colors=class_colors,
                        log_batch_frequency=log_batch_frequency,
                        log_best_prefix=log_best_prefix)
        self.tf_Dataset = generator #TF dataset

        self.sample_data, self.validation_data = self.setup_validation_data(
            generator, predictions, batch_size, validation_steps
        )
        # generator for evaluation metrics ()
        self.generator  = None

        # flags
        self.log_confusion_matrix = log_confusion_matrix
        self.log_predictions      = log_predictions
        self.log_uncertainty      = log_uncertainty
        self.log_metrics          = log_metrics
        self.log_roc_curve        = log_roc_curve

    def setup_validation_data(self, generator, predictions, 
                              batch_size, validation_steps, sample_size=None):
        # predictions
        take_predictions = int(np.ceil(predictions/batch_size))
        take_predictions = min(validation_steps, take_predictions)
        take_predictions = np.random.choice(np.arange(validation_steps), 
                                size=take_predictions, replace=False)
        # samples
        sample_size = sample_size or int(predictions*0.2)
        take_samples = int(np.ceil(sample_size/batch_size))
        take_samples = min(validation_steps, take_samples)
        take_samples = np.random.choice(np.arange(validation_steps), 
                                size=take_samples, replace=False)
        x_test, y_test = None, None
        x_samples, y_samples = None, None
        for i, (batch_x, batch_y) in enumerate(generator):
            (batch_x, batch_y) = (np.array(batch_x), np.array(batch_y))
            if i in take_samples:
                if x_samples is None:   # first iteration
                    x_samples, y_samples = (batch_x, batch_y)
                else:
                    x_samples, y_samples = (
                        np.vstack((x_samples, batch_x)), 
                        np.vstack((y_samples, batch_y))
                    )
            if i in take_predictions:
                if x_test is None:   # first iteration
                    x_test, y_test = (batch_x, batch_y)
                else:
                    x_test, y_test = (
                        np.vstack((x_test, batch_x)), 
                        np.vstack((y_test, batch_y))
                    )
            if i+1 == validation_steps: break
        # TODO: check lenght of data
        sample_data = (x_samples[:sample_size], y_samples[:sample_size])
        validation_data = (x_test, y_test)
        return sample_data, validation_data

    def on_epoch_end(self, epoch, logs={}):
        #self.generator  = self.tf_Dataset.as_numpy_iterator()
        _best = {}
        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        if any((self.log_predictions, self.log_uncertainty, 
                self.log_confusion_matrix, self.log_metrics)):
            validation_data, predictions_uncertainty = self._get_predictions_uncertainty()

        if self.log_metrics:
            (_, y_true) = self._get_data(mode="labels")
            _, y_pred, _ = predictions_uncertainty
            _metrics = self._log_metrics(y_pred, y_true)
            class_stat_table, overall_stat_table, overall_and_class_values = _metrics 
            wandb.log(overall_and_class_values, commit=False)
            wandb.log({
                "Class Stat": class_stat_table,
                "Overall Stat": overall_stat_table
            }, commit=False)
            for k,v in overall_and_class_values.items(): _best[k] = v
            _best["Class Stat"] = class_stat_table
            _best["Overall Stat"] = overall_stat_table

        if self.log_roc_curve:
            (_, y_true) = self._get_data(mode="labels")
            y_predictive_distribution, _, _ = predictions_uncertainty
            roc_plot = self._log_roc_curve(y_predictive_distribution, y_true)
            wandb.log(
               { "ROC Curve": roc_plot}, commit=False
            )
            _best["ROC Curve"] = roc_plot

        if self.log_confusion_matrix:
            _, y_true = self._get_data(mode="labels")
            _, y_pred, _ = predictions_uncertainty
            cm_plot = self._log_confusion_matrix(y_pred, y_true)
            wandb.log(
               { "Confusion Matrix": cm_plot}, commit=False
            )
            _best["Confusion Matrix"] = cm_plot

        if self.log_predictions:
            y_predictive_distribution, y_pred, _ = predictions_uncertainty
            examples = self._log_predictions(
                *validation_data, y_predictive_distribution, y_pred
            )
            wandb.log(
                {"Examples": examples},
               commit=False
            )
            _best["Examples"] = examples

        if self.log_uncertainty:
            y_predictive_distribution, _, y_predictions_samples = predictions_uncertainty
            uncertainty = self.model.uncertainty(
                y_predictive_distribution=y_predictive_distribution,
                y_predictions_samples=y_predictions_samples
            )
            _mean_uncertainty = self._log_mean_uncertainty(uncertainty)
            wandb.log(
                _mean_uncertainty,
                commit=False
            ) 
            _uncertainty = {
                "Uncertainty": self._log_epistemic_uncertainty(
                    *validation_data, 
                    *predictions_uncertainty,
                    uncertainty
                ),
                "Uncertainty By Class": self._log_uncertainty_by_class(
                    validation_data[1], 
                    uncertainty
                ),
                "High Uncertainty": self._log_epistemic_uncertainty_highlights(
                    *validation_data,
                    *predictions_uncertainty,
                    uncertainty, mode='max'
                ),
                "Low Uncertainty": self._log_epistemic_uncertainty_highlights(
                    *validation_data,
                    *predictions_uncertainty,
                    uncertainty, mode='min'
                )
            }
            wandb.log(
                _uncertainty,
               commit=False
            )
            for k,v in _mean_uncertainty.items(): _best[k] = v
            for k,v in _uncertainty.items(): _best[k] = v
        wandb.log({"epoch": epoch}, commit=False)
        wandb.log(logs, commit=True)

        # Save best performance
        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary[
                    "%s%s" % (self.log_best_prefix, self.monitor)
                ] = self.current
                wandb.run.summary[
                    "%s%s" % (self.log_best_prefix, "epoch")
                ] = epoch
                for k,v in _best.items():
                    wandb.run.summary[
                        "%s%s" % (self.log_best_prefix, k)
                    ] = v 
                if self.verbose and not self.save_model:
                    print(
                        "Epoch %05d: %s improved from %0.5f to %0.5f"
                        % (epoch, self.monitor, self.best, self.current)
                    )
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current

    def _get_data(self, mode="onehot"):
        X_test, y_test = self.validation_data
        if mode == "labels":
            y_test = y_test.argmax(axis=1)
        elif mode == "onehot":
            pass
        else:
            raise ValueError("Invalid data mode:", mode)
        return X_test, y_test

    def _get_predictions_uncertainty(self):
        X_test, y_test = self._get_data(mode="labels")
        predictions = self.model.predict_distribution(x=X_test)
        y_predictive_distribution, y_pred, y_predictions_samples = predictions
        return  (X_test, y_test),\
                (y_predictive_distribution, y_pred, y_predictions_samples)



class AleatoricCallback(EpistemicCallback):
    """
    WandbCallback extension for include extra plots and summaries for an aleatoric model (AleatoricModel).
    """
    def on_epoch_end(self, epoch, logs={}):
        #self.generator  = self.tf_Dataset.as_numpy_iterator()
        _best = {}
        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        if any((self.log_predictions, self.log_uncertainty, 
                self.log_confusion_matrix, self.log_metrics)):
            validation_data, predictions_uncertainty = self._get_predictions_uncertainty()

        if self.log_metrics:
            (_, y_true) = self._get_data(mode="labels")
            _, y_pred, _ = predictions_uncertainty
            _metrics = self._log_metrics(y_pred, y_true)
            class_stat_table, overall_stat_table, overall_and_class_values = _metrics 
            wandb.log(overall_and_class_values, commit=False)
            wandb.log({
                "Class Stat": class_stat_table,
                "Overall Stat": overall_stat_table
            }, commit=False)
            for k,v in overall_and_class_values.items(): _best[k] = v
            _best["Class Stat"] = class_stat_table
            _best["Overall Stat"] = overall_stat_table

        if self.log_roc_curve:
            (_, y_true) = self._get_data(mode="labels")
            y_predictive_distribution, _, _ = predictions_uncertainty
            roc_plot = self._log_roc_curve(y_predictive_distribution, y_true)
            wandb.log(
               { "ROC Curve": roc_plot}, commit=False
            )
            _best["ROC Curve"] = roc_plot

        if self.log_confusion_matrix:
            _, y_true = self._get_data(mode="labels")
            _, y_pred, _ = predictions_uncertainty
            cm_plot = self._log_confusion_matrix(y_pred, y_true)
            wandb.log(
               { "Confusion Matrix": cm_plot}, commit=False
            )
            _best["Confusion Matrix"] = cm_plot

        if self.log_predictions:
            y_predictive_distribution, y_pred, _ = predictions_uncertainty
            examples = self._log_predictions(
                *validation_data, y_predictive_distribution, y_pred
            )
            wandb.log(
                {"Examples": examples},
                commit=False
            )
            _best["Examples"] = examples

        if self.log_uncertainty:
            y_predictive_distribution, y_pred, y_predictive_variance = predictions_uncertainty
            uncertainty = self.model.uncertainty(y_predictive_variance=y_predictive_variance)
            _mean_uncertainty = self._log_mean_uncertainty(uncertainty)
            wandb.log(
                _mean_uncertainty,
                commit=False
            ) 
            _uncertainty = {
                    "Uncertainty": self._log_aleatoric_uncertainty(
                        *validation_data, 
                        y_predictive_distribution,
                        y_pred,
                        uncertainty
                    ),
                    "Uncertainty By Class": self._log_uncertainty_by_class(
                        validation_data[1], 
                        uncertainty
                    ),
                    "High Uncertainty": self._log_aleatoric_uncertainty_highlights(
                        *validation_data,
                        y_predictive_distribution,
                        y_pred,
                        uncertainty, mode='max'
                    ),
                    "Low Uncertainty": self._log_aleatoric_uncertainty_highlights(
                        *validation_data,
                        y_predictive_distribution,
                        y_pred,
                        uncertainty, mode='min'
                    )
            }
            wandb.log(
                _uncertainty,
                commit=False
            )
        wandb.log({"epoch": epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary[
                    "%s%s" % (self.log_best_prefix, self.monitor)
                ] = self.current
                wandb.run.summary[
                    "%s%s" % (self.log_best_prefix, "epoch")
                ] = epoch
                for k,v in _best.items():
                    wandb.run.summary[
                        "%s%s" % (self.log_best_prefix, k)
                    ] = v 
                if self.verbose and not self.save_model:
                    print(
                        "Epoch %05d: %s improved from %0.5f to %0.5f"
                        % (epoch, self.monitor, self.best, self.current)
                    )
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current

    def _get_predictions_uncertainty(self):
        X_test, y_test = self._get_data(mode="labels")
        predictions = self.model.predict_variance(x=X_test)
        y_predictive_distribution, y_pred, y_predictions_variance = predictions
        return  (X_test, y_test),\
                (y_predictive_distribution, y_pred, y_predictions_variance)    


def setup_evaluation_logger( model_name, enable_wandb=True, labels=None, max_plots=150,
                   classification_metrics={}, aleatoric_uncertainty={},
                   aggregation_metrics={}, run_dir="."):
    if enable_wandb:
        return EvaluationLogger(model_name, labels, 
            classification_metrics=classification_metrics, 
            aggregation_metrics=aggregation_metrics, 
            aleatoric_uncertainty=aleatoric_uncertainty
        )
    else:
        # This can be implemented later ;)
        run_dir=run_dir
        raise NotImplementedError()

class EvaluationLogger(WandBLogGenerator):
    def __init__(self, model_name, labels, max_plots=150,
                 classification_metrics={}, 
                 aleatoric_uncertainty={}, 
                 aggregation_metrics={}
        ):
        super().__init__(model_name, labels, max_plots=max_plots)
        self._classification_metrics = {
            "log_predictions": classification_metrics.get("log_predictions", False), 
            "log_uncertainty": classification_metrics.get("log_uncertainty", True), # prediction + uncertainty
            "log_class_uncertainty": classification_metrics.get("log_class_uncertainty", True),
            "log_ranking_uncertainty": classification_metrics.get("log_ranking_uncertainty", True),
            "log_confusion_matrix": classification_metrics.get("log_confusion_matrix", True), 
            "log_roc_curve": classification_metrics.get("log_roc_curve", True), 
            "log_metrics": classification_metrics.get("log_metrics", True)
        }
        self._aleatoric_uncertainty = {
            "log_class_uncertainty": aleatoric_uncertainty.get("log_class_uncertainty", True),
            "log_ranking_uncertainty": aleatoric_uncertainty.get("log_ranking_uncertainty", True)
        }
        self._aggregation_metrics = {
            "log_prediction": aggregation_metrics.get("log_prediction", True),
            "log_prediction_map": aggregation_metrics.get("log_prediction_map", True),
            "log_uncertainty_map": aggregation_metrics.get("log_uncertainty_map", True),
            "log_confusion_matrix": aggregation_metrics.get("log_confusion_matrix", True),
            "log_ranking_uncertainty": aggregation_metrics.get("log_ranking_uncertainty", True),
            "log_class_uncertainty": aggregation_metrics.get("log_class_uncertainty", True),
            "log_roc_curve": aggregation_metrics.get("log_roc_curve", True),
            "log_metrics": aggregation_metrics.get("log_metrics", True),
        }

    def log_aleatoric_uncerainty(self):
        pass

    def log_classification_metrics(self, X, y_true, y_pred, y_predictive_distribution, 
                  y_predictions_samples, uncertainty):
        log_configuration = self._classification_metrics
        uncertainty = pd.DataFrame(uncertainty) if isinstance(uncertainty, dict) else uncertainty
        if log_configuration["log_metrics"]:
            _metrics = self._log_metrics(y_pred, y_true)
            class_stat_table, overall_stat_table, overall_and_class_values = _metrics 
            wandb.log(overall_and_class_values, commit=False)
            wandb.log({
                "Class Stat": class_stat_table,
                "Overall Stat": overall_stat_table
            }, commit=False)
        if log_configuration["log_roc_curve"]:
            roc_plot = self._log_roc_curve(y_predictive_distribution, y_true)
            wandb.log({ "ROC Curve": roc_plot}, commit=False)

        if log_configuration["log_confusion_matrix"]:
            cm_plot = self._log_confusion_matrix(y_pred, y_true)
            wandb.log({"Confusion Matrix": cm_plot}, commit=False)

        if log_configuration["log_predictions"]:
            wandb.log(
                {"Examples": self._log_predictions(
                        X, y_true, y_predictive_distribution, y_pred
                    )},
                commit=False
            )

        if log_configuration["log_uncertainty"]:
            wandb.log(
                self._log_mean_uncertainty(uncertainty),
                commit=False
            ) 
            wandb.log(
                {
                    "Uncertainty": self._log_epistemic_uncertainty(
                        X, y_true, 
                        y_predictive_distribution, y_pred, 
                        y_predictions_samples, uncertainty
                    ),
                    "Uncertainty By Class": self._log_uncertainty_by_class(
                        y_true, 
                        uncertainty
                    ),
                    "High Uncertainty": self._log_epistemic_uncertainty_highlights(
                        X, y_true, 
                        y_predictive_distribution, y_pred, 
                        y_predictions_samples, uncertainty,
                        mode='max'
                    ),
                    "Low Uncertainty": self._log_epistemic_uncertainty_highlights(
                        X, y_true, 
                        y_predictive_distribution, y_pred, 
                        y_predictions_samples, uncertainty,
                        mode='min'
                    )
                },
               commit=False
            )
        wandb.log({}, commit=True)

    def log_aggregation_metrics(self):
        pass