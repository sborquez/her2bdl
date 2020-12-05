"""
Utility tools for train/evaluate experiments
================================================

Collection of functions shared between train scripts.
"""

import os
from os.path import join
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from .visualization.metrics import display_confusion_matrix
from .visualization.prediction import (
    display_prediction, display_uncertainty, display_uncertainty_by_class
)

__all__ = [
    "load_config_file", "load_run_config",
    "setup_callbacks", "setup_experiment"
]


# Experiment files variables
__sections = set([
    "experiment", "model", "aggregation", "data",
    "training", "evaluate", "predict", "plugins",
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
    "evaluate":
        ["metrics"],
    "predict":
        ["save_aggregation", "save_predictions", "save_uncertainty"]
}

__default_optional = {
    "experiment":
        {"tags": [], "run_id": None, "seed": 1234},
    "model":
        {"weights": None, "seed": 1234},
    "aggregation":
        {},
    "data":
        {"preprocessing": {}, "label_mode": "categorical", "labels": None},
    "training":
        {"batch_size": 16, "validation_split": 0.2,
         "optimizer":
            {"name": "adam", "learning_rate": 0.4, "parameters": None},
        },
    "evaluate":
        {},
    "predict":
        {},
    "plugins":
        {"wandb": None}
}

def parse_wandb_config(config):
    return {
        section: config.get(section)["value"]
        for section in __sections if section in (set(config.keys()) - set(("wandb_version", "_wandb")))
    }

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
            experiments_folder = \
                experiment_config['experiment']['experiments_folder']
            run_id = experiment_config["experiment"]["run_id"]
            folder_ = f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{run_id}"
            experiment_folder = join(experiments_folder, folder_)
            os.makedirs(experiment_folder, exist_ok=True)
            with open(join(experiment_folder, 'config.yaml'), 'w') as outfile:
                yaml.dump(experiment_config, outfile)
            return experiment_folder
        else:
            raise NotImplementedError


def setup_callbacks(validation_data, validation_steps, model_name, batch_size, 
    	           enable_wandb, labels=None,
                   earlystop=None, experiment_tracker=None, checkpoints=None,
                   run_dir="."):
    """
    Callbacks configuration:
        # Early stop, use null to disable.
        earlystop:
            patience: 15
            monitor: val_loss
        # Experiments results while training
        experiment_tracker:
            # Save model architecture summary
            log_model_summary: true
            # Save datasets info: source, sizes, etc.
            log_dataset_description: true
            # Plots and logs
            log_training_loss_plot: true
            log_training_loss: true
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
        # TODO use experiment_tracker parameters    
        callbacks.append(
            UncertantyCallback(
                generator=validation_data,
                validation_steps=len(validation_data),
                model_name=model_name,
                batch_size=batch_size,
                labels=labels
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


class UncertantyCallback(wandb.keras.WandbCallback):
    """
    WandbCallback extension for include extra plots and summaries.
    """

    def __init__(self, generator, validation_steps, model_name, batch_size,
                 monitor='val_loss', verbose=0, mode='auto',
                 save_weights_only=True, log_weights=True, log_gradients=False,
                 save_model=True, training_data=None, validation_data=None,
                 labels=None, data_type=None, predictions=200, input_type="image",
                 output_type="label", log_evaluation=False, class_colors=None,
                 log_batch_frequency=None, log_best_prefix="best_",
                 log_confusion_matrix=True, log_predictions=True, 
                 log_uncertainty=True):
        super().__init__(monitor=monitor,
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
        self.model_name = model_name

        self.sample_data, self.validation_data = self.setup_validation_data(
            generator, predictions, batch_size, validation_steps
        )
        # generator for evaluation metrics ()
        self.generator  = None

        # flags
        self.max_plots            = 35
        self.log_confusion_matrix = log_confusion_matrix
        self.log_predictions      = log_predictions
        self.log_uncertainty      = log_uncertainty

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
        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        if any((self.log_predictions, self.log_uncertainty, self.log_confusion_matrix)):
            validation_data, predictions_uncertainty = self._get_predictions_uncertainty()

        if self.log_confusion_matrix:
            #(x_true, y_true, y_pred)  = self._get_predictions()
            (x_true, y_true) = validation_data
            _, y_pred, _ = predictions_uncertainty
            wandb.log(
               {"Confusion Matrix": \
                   self._log_confusion_matrix(y_pred, y_true)},
               commit=False
            )

        if self.log_predictions:
            y_predictive_distribution, y_pred, _ = predictions_uncertainty
            wandb.log(
                {"Examples": self._log_predictions(
                   *validation_data, 
                    y_predictive_distribution, y_pred
                )},
               commit=False
            )
        if self.log_uncertainty:
            y_predictive_distribution, _, y_predictions_samples = predictions_uncertainty
            uncertainty = self.model.uncertainty(
                y_predictive_distribution=y_predictive_distribution,
                y_predictions_samples=y_predictions_samples
            )

            wandb.log(
                self._log_mean_uncertainty(uncertainty),
                commit=False
            ) 
            wandb.log(
                {
                    "Uncertainty": self._log_uncertainty(
                        *validation_data, 
                        *predictions_uncertainty,
                        uncertainty
                    ),
                    "Uncertainty By Class": self._log_uncertainty_by_class(
                        validation_data[1], 
                        uncertainty
                    ),
                    "High Uncertainty": self._log_uncertainty_highlights(
                        *validation_data,
                        *predictions_uncertainty,
                        uncertainty, mode='max'
                    ),
                    "Low Uncertainty": self._log_uncertainty_highlights(
                        *validation_data,
                        *predictions_uncertainty,
                        uncertainty, mode='min'
                    )
                },
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
                if self.verbose and not self.save_model:
                    print(
                        "Epoch %05d: %s improved from %0.5f to %0.5f"
                        % (epoch, self.monitor, self.best, self.current)
                    )
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current

    def _get_predictions(self):
        X_test, y_test = self.validation_data
        y_test = y_test.argmax(axis=1)
        y_pred = self.model.predict(X_test).argmax(axis=1)
        return  (X_test, y_test, y_pred)

    def _get_predictions_uncertainty(self):
        #X_sample, y_sample = self.sample_data
        X_sample, y_sample = self.validation_data
        y_sample = y_sample.argmax(axis=1)
        predictions = self.model.predict_distribution(x=X_sample)
        y_predictive_distribution, y_pred, y_predictions_samples = predictions
        return  (X_sample, y_sample),\
                (y_predictive_distribution, y_pred, y_predictions_samples)

    def _log_confusion_matrix(self, y_pred, y_true):
        """
        Display a deterministic confusion matrix evaluating validation_data.
        Parameters
        ----------
        Return
        ------
        """
        figure = display_confusion_matrix(
                    y_true=y_true, y_pred=y_pred,
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
        Return
        ------
        """
        wandb_images = []
        length = len(x_true)
        if length > self.max_plots :
            choices = np.random.choice(range(length), 
                                      size=self.max_plots , replace=False)
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
        mean = dict(uncertainty.mean())
        return {f"mean_{k}":v for k,v in mean.items()}

    def _log_uncertainty(self, x_true, y_true, 
                         y_predictive_distribution, y_pred, 
                         y_predictions_samples, uncertainty):
        """
        Display sample and models uncertainty.
        Parameters
        ----------
        Return
        ------
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

    def _log_uncertainty_highlights(self, x_true, y_true, 
                                    y_predictive_distribution, y_pred, 
                                    y_predictions_samples, uncertainty, 
                                    mode='max'):
        """
        Display sample and models uncertainty.
        Parameters
        ----------
        Return
        ------
        """
        highlights = []
        for metric in uncertainty:
            if mode == 'max':
                index = uncertainty[metric].argmax()
            elif mode == 'min':
                index = uncertainty[metric].argmin()
            highlights.append((index, metric))
        caption = "higher" if mode == "max" else "lower"
        wandb_images = []
        #for i in range(len(x_true)):
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


def setup_evaluation_logger(test_generator, model_name, batch_size, enable_wandb, labels=None, run_dir="."):
    if enable_wandb:
        pass
    else:
        pass

class EvaluationLogger():
    def __init__(self, test_generator):
        pass

    def log(self, evaluation, predictive_distribution):
        pass

    def _log_uncertainty_map(self):
        pass

class EvaluationWandBLogger(EvaluationLogger):
    def __init__(self, test_generator):
        pass

    def log(self, evaluation, predictive_distribution):
        pass

    def _log_uncertainty_map(self):
        pass