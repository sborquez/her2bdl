"""
Utility tools for train/evaluate experiments
================================================

Collection of functions shared between train scripts.
"""

import os
from datetime import datetime
import numpy as np
import yaml
import wandb
from wandb.keras import WandbCallback
    
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from .visualization.metrics import display_confusion_matrix 

__all__ = ["load_config_file", "setup_callbacks", "setup_experiment"]


# Experiment files variables
__sections = set([
    "experiment", "model", "aggregation", "data",
    "training", "evaluate", "predict", "plugins"
])

__required = {
    "experiment": 
        ["name", "experiments_folder"],
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
        ["save_aggregation", "save_predictions", "save_uncertainty"],
    "plugins":
        []
}

__default_optional = {
    "experiment": 
        {"run_id": None, "seed": 1234},
    "model": 
        {"weights": None, "seed": 1234},
    "aggregation":
        {},
    "data":
        {"preprocessing": {}, "label_mode": "categorical"},
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


def load_config_file(config_filepath):
    """Load, validate and set default values"""
    # Read config file
    with open(config_filepath) as yaml_file:
        base_config = yaml.safe_load(yaml_file)
    # Check sections
    assert set(base_config.keys()) == __sections, "Configuration file has missings sections."
    # Check required sections and subsections
    not_defined = {}
    for section, required_keys in __required.items():
        defined_keys = base_config[section].keys()
        if len(set(required_keys) - set(defined_keys)) > 0:
            not_defined[section] = set(required_keys) - set(defined_keys)
    assert len(not_defined) == 0, f"Configuration not defined: {not_defined}"
    # Default values
    config = dict(__default_optional)
    for section in __default_optional.keys():
        config[section].update(base_config[section])
    # Set Experiment run id
    if config["experiment"]["run_id"] is None:
        config["experiment"]["run_id"] = wandb.util.generate_id()
    return config

#config = load_config_file("D:/sebas/Google Drive/Projects/her2bdl/train/experiments/config/simple_classifier.yaml")
def setup_experiment(experiment_config, mode="training"):
    # Weight and Bias
    plugins = experiment_config.get("plugins", {})
    del experiment_config["plugins"]
    wand_config = plugins.get("wandb", None)
    if wand_config is not None:
        assert wand_config["apikey"] is not None, "Requiere enviroment variable 'WANDB_API_KEY' or API Key."
        if wand_config["apikey"]  == 'WANDB_API_KEY':
            assert wand_config["apikey"] in os.environ, "Requiere enviroment variable 'WANDB_API_KEY' defined." 
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
    else:
        if mode == "training":
            experiment_config["training"]["callbacks"]["enable_wandb"] = False # check consistency
            experiments_folder = experiment_config['experiment']['experiments_folder']
            run_id = experiment_config["experiment"]["run_id"]
            experiment_folder   = os.path.join(experiments_folder, f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{run_id}")
            os.makedirs(experiment_folder, exist_ok=True)
            with open(os.path.join(experiment_folder, 'config.yml'), 'w') as outfile:
                yaml.dump(experiment_config, outfile)
        else:
            raise NotImplementedError


def setup_callbacks(validation_data, validation_steps, model_name, batch_size, enable_wandb, earlystop=None, experiment_tracker=None, checkpoints=None):
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
        callbacks = [
            UncertantyCallback(
                generator=validation_data,
                validation_steps=len(validation_data),
                model_name=model_name,
                batch_size=batch_size,
                #labels=list(range(10))
            )
        ]
    else:
        pass
    return callbacks


class UncertantyCallback(wandb.keras.WandbCallback):
    """
    WandbCallback extension for include extra plots and summaries.

    original from: https://wandb.ai/mathisfederico/wandb_features/reports/Visualizing-Confusion-Matrices-With-W-B--VmlldzoxMzE5ODk
    """

    def __init__(self, generator, validation_steps, model_name, batch_size, monitor='val_loss', verbose=0, mode='auto',
                 save_weights_only=True, log_weights=True, log_gradients=False,
                 save_model=True, training_data=None, validation_data=None,
                 labels=[], data_type=None, predictions=25,
                 input_type="image", output_type="label", log_evaluation=False, class_colors=None, log_batch_frequency=None,
                 log_best_prefix="best_", log_confusion_matrix=True):
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
        # take random samples for plots
        take = int(np.ceil(predictions/batch_size))
        take = min(validation_steps, take)
        take = np.random.choice(np.arange(validation_steps), size=take, replace=False)
        x_val, y_val = None, None
        for i, (batch_x, batch_y) in enumerate(self.tf_Dataset):
            if i in take:
                if x_val is None:   # first iteration
                    x_val, y_val = (batch_x, batch_y)
                else:
                    x_val, y_val = (np.vstack((x_val, batch_x)), np.vstack((y_val, batch_y)))
            if i+1 == validation_steps: break

        self.validation_data = (
            x_val, y_val
        )
        # generator for evaluation metrics ()
        self.generator  = None

        # flags
        self.log_confusion_matrix = log_confusion_matrix

    def on_epoch_end(self, epoch, logs={}):
        self.generator  = self.tf_Dataset.as_numpy_iterator()
        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        if self.input_type in ("image", "images"):
            wandb.log(
                {"examples": self._log_images(num_images=self.predictions)},
                commit=False,
            )

        if self.log_confusion_matrix:
            wandb.log(
                {"Confusion Matrix": self._log_confusion_matrix()}
            )

        wandb.log({"epoch": epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary[
                    "%s%s" % (self.log_best_prefix, self.monitor)
                ] = self.current
                wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
                if self.verbose and not self.save_model:
                    print(
                        "Epoch %05d: %s improved from %0.5f to %0.5f"
                        % (epoch, self.monitor, self.best, self.current)
                    )
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current

    def _log_confusion_matrix(self):
        """
        Display a deterministic confusion matrix evaluating validation_data.
        """
        X_test, y_test = self.validation_data
        y_test = y_test.argmax(axis=1)
        y_pred = self.model.predict(X_test).argmax(axis=1)
        figure = display_confusion_matrix(y_test, y_pred,
                    model_name=self.model_name,
                    target_names=list(map(str, np.arange(10)))
        )
        img_log = wandb.Image(figure)
        plt.close(figure)
        return img_log