# Experiments Configuration Files

Experiments files are shareable experiment configurations writed in `YAML` format.

These file is loaded by `train_model.py` and `evaluate.py` scripts, loading the experiment configuration and setup the training or evalution environment.

## Configuration file structure

An experiment configuration file is divided in 7 sections, a valid configuration file require this sections, but some parameters are optional. 
### `experiment`
 Information for experiment identification. This also includes the `experiment_folder`, path where files are saved. This section contains the following parameters.

  * `name`: Human readable experiment's name.
  * `notes`: Human redeable small experiment`s description.
  * `experiment_folder`: Parent folder for this experiment. (This folder may contains multiples experiments)
  * `tags`: Keywords,indetification tags. (optional, default=`[]`)
  * `run_id`: Numeric identfier for the experiment. If `null`, will be randomly picked. (optional, default=`null`)
  * `seed`: Random seed, if it's null, uses a random seed. (optional, default=`null`) 
### `model`
Model architecture and uncertainty parameters.
  
  * `task`: Model's task.
  * `architecture`: Model's base architecture, defined in `her2bdl/her2bdl/models/mcdropout.py`.
  * `weights`: Models pretrained checkpoint or weights path. If it is null, use random weight initialization. (optional, default=`null`)
  * `hyperparameters`: Architecture hyperparameters, e.g. `mc_dropout_rate`. Each implemented architecture required differents hyperparameters.
  * `uncertainty`: Model uncertainty hyperparameters. These hyperparameters doesn't affect model architecture, but influence uncertainty estimation.
### `aggregation`
Aggregation methods for aggregation multilples outputs for one input.

  * `method`: Base methods for aggreation.
  * `parameters`: Aggregation methods parameters. They may vary for each method.
### `data`
Datasets and preprocessing.

  * `source`: Data source, more details in the following section (_Data sources_).
  * `img_height`: Input image heigh, this depends on model architecture.
  * `img_width`: Input image width, this depends on model architecture.
  * `img_channels`: Input image channels, this depends on model architecture.
  * `preprocessing`: Data preprocessing steps: `rescale` is used for rescale images values. `aggregate_dataset_parameters` are parameters for `aggregate_dataset` function.
  * `num_classes`: Number of classes.
  * `label_mode`: Depend on the tasks. e.g. _categorical_.
  * `labels`: List of label names.
  

### `training`

Training model hyperparameters and callbacks.

* `epochs`: Training epochs.
* `batch_size`: Depends on CPU-GPU available memory. (optional, default=16) 
* `validation_split`: Train and validation split. (Optional, default=0.2)
* `loss`: Model loss
  
  *  `function`: Loss function builder.
  *  `parameters`: Loss function parameters.
* `optimizer`: Model optimizer.
  
  *  `name`: Optimizer name.
  *  `learning_rate`: Optimizer learning rate.
  *  `parameters`: Optimizer function parameters.
* `callbacks`: Callback options. Currently supporting `checkpoints`, `early_stop` and `experiment_tracker` custom callback. Also include `enable_wandb` option. (check plugins).

### `evaluate`

Evaluation metrics. 

* `metrics`: List of keras metrics.
### `predict`

Prediction options. TBA

### `plugins`

Plugin setups, currently supporting **Weight and Bias** experiment tracking platform.

* `wandb`: this required the project's name (`project`) and `apikey`. The api-key can be set to '_WANDB_API_KEY_' if the api-key is stored as a enviroment variable. Set `wandb`to `null` to disable it.


## Data sources

Dataset can be loaded from two sources:
#### Tensorflow Datasets (`'tf_Dataset'`)

These datasets are used for testing purposes.

`'tf_Dataset'` requires the `dataset_target` parameter. Check `her2bdl/data/generator.py:get_generators_from_tf_Dataset` for a list of dataset options.

**Example:** This is an example for a configuration file that uses tensorflow datasets (`test_simple_classifier.json`)

```yaml
# Experiment information
experiment:
  # Custom indentifier name for the experiment.
  # Experiment
  name: testing_model
  notes: simple test for trainig a classifier.
  tags: 
    - test
    - simple
  # Experiments folder: Contains multiples experiments
  experiments_folder: D:/sebas/Google Drive/Projects/her2bdl/train/experiments/runs
  # experiments_folder: /data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl/train/experiments/runs
  # (Optional, default=null) numeric identfier for the experiment. If `null`, will be randomly picked.
  run_id: null
  # (Optional, default=null) Random seed, if it`s null, uses a random seed.
  seed: 1234

# Model architecture
model:
  # Model task: [classification|regression|hierarchical]
  task: classification
  # Model Class implemented in her2bdl.models
  architecture: SimpleClassifierMCDropout
  # (Optional, default=null) Models pretrained weights path. If it is null, use random weight initialization.
  weights: null
  # Model architecture hyperparameters
  hyperparameters: 
    mc_dropout_rate: 0.5
  # Model uncertainty hyperparameters
  uncertainty:
    sample_size: 100
    mc_dropout_batch_size: 16
    multual_information: true
    variation_ratio: true
    predictive_entropy: true

# Aggregation methods
aggregation: 
  # Method Class
  method: null
  # Mathod parameters
  parameters: null

# Datasets and preprocessing
data:
  # Source
  source:
    type: tf_Dataset
    # Source type parameters
    parameters:
      dataset_target: simple
      data_dir: null
  # Input
  img_height: 224
  img_width: 224
  img_channels: 3
  preprocessing: 
    rescale: 0.00392156862745098
  # Target
  num_classes: 10
  label_mode: categorical  # Depends on task and model architecture
  labels: 
    - '0'
    - '1'
    - '2'
    - '3'
    - '4'
    - '5'
    - '6'
    - '7'
    - '8'
    - '9'

# Training model hyperparameters
training:
  # Training epochs
  epochs: 5
  # (Optional, default=16) Depends on CPU-GPU available memory.
  batch_size: 64 
  # (Optional, default=0.2) Train and validation split.
  validation_split: "sample"
  # Models loss function
  loss:
    function: CategoricalCrossentropy
    parameters: null
  # (Optional, default=Adam(lr=1e-4)) Training optimizer
  optimizer:
    name: adam
    learning_rate: 1e-4
    parameters: null
  # Training callbacks
  callbacks:
    # Connect to weight and bias *Requiere plugins and env variable.* 
    enable_wandb: true
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
# Evaluation metrics
evaluation:
  metrics: [accuracy]
# Predict 
predict:
  save_aggregation: false
  save_predictions: true
  save_uncertainty: false
# Plugin setups
plugins:
  wandb:
    project: Her2BDL
    # Use environment variable name (recommended) or API KEY.
    #apikey: /data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl/.wandb_secret
    apikey: /home/asuka/projects/her2bdl/.wandb_secret
```


#### HER2 WSI datasets (`'wsi'`)

For HER2 classification, this load a preprocessed version of the Warwick HER2 dataset. Check preprocesing (TBA) for more information.

`'wsi'` requires parameters for `train_generator`, `validation_generator` and `test_generator`:

* `generator`: '_GridPatchGenerator_' or '_MCPatchGenerator_'
* `generator_parameters`:
    
    * `dataset`: Path to `.csv` dataset file.
    * `patch_level`: WSI level for patches.
    * `patch_vertical_flip`: Flag for image augmentation.
    * `patch_horizontal_flip`: Flag for image augmentation.
    * `shuffle`: Flag for shuffle dataset at every epoch end.



**Example:** This is an example of using HER2 WSI datasets (`wsi_generators/efficientnet_b0_her2_grid.yaml`):


```yaml
# Experiment information
experiment:
  # Custom indentifier name for the experiment.
  # Experiment
  name: EfficientNetB0-McDropout C876 GG
  notes: EfficientNet with MC-Dropout for Her2 Classification using WSI-GridPatchGenerator.
  tags:
    - efficientnet
    - b0
    - her2
    - mcdropout
    - training
    - GridGenerator
  # Experiments folder: Contains multiples experiments
  experiments_folder: D:/sebas/Projects/her2bdl/train/experiments/runs
  #experiments_folder: /data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl/train/experiments/runs
  # (Optional, default=null) numeric identfier for the experiment. If `null`, will be randomly picked.
  run_id: null
  # (Optional, default=null) Random seed, if it`s null, uses a random seed.
  seed: 1234

# Model architecture
model:
  # Model task: [classification|regression]
  task: classification
  # Model Class implemented in her2bdl.models
  architecture: EfficientNetMCDropout
  # (Optional, default=null) Models pretrained weights path. If it is null, use random weight initialization.
  weights: null
  # Model architecture hyperparameters
  hyperparameters: 
    mc_dropout_rate: 0.5
    base_model: B0
    efficient_net_weights: imagenet
    classifier_dense_layers: 
      - 256
      - 128
      - 64
  # Model uncertainty hyperparameters
  uncertainty:
    sample_size: 200
    mc_dropout_batch_size: 32
    multual_information: true
    variation_ratio: true
    predictive_entropy: true

# Aggregation methods
aggregation: 
  # Method Class
  method: null
  # Mathod parameters
  parameters: null

# Datasets and preprocessing
data:
  # Source
  source:
    type: wsi
    # Source type parameters
    parameters:
      train_generator:
        generator: 'GridPatchGenerator'
        generator_parameters:
          #dataset: '/'
          dataset: 'D:/sebas/Projects/her2bdl/train/datasets/train.csv'
          patch_level: 3
          patch_vertical_flip: false
          patch_horizontal_flip: false
          shuffle: true
      validation_generator:
        generator: 'GridPatchGenerator'
        generator_parameters:
          dataset: 'D:/sebas/Projects/her2bdl/train/datasets/validation.csv'
          patch_level: 3
          patch_vertical_flip: false
          patch_horizontal_flip: false
          shuffle: true
      test_generator:
        generator: 'GridPatchGenerator'
        generator_parameters:
          dataset: 'D:/sebas/Projects/her2bdl/train/datasets/test.csv'
          patch_level: 3
          patch_vertical_flip: false
          patch_horizontal_flip: false
          shuffle: true
  # Input
  img_height: 224
  img_width: 224
  img_channels: 3
  preprocessing:
    rescale: null
    aggregate_dataset_parameters: null
  # Target
  num_classes: 4
  label_mode: categorical  # Depends on task and model architecture
  labels: 
    - '0'
    - '1+'
    - '2+'
    - '3+'

# Training model hyperparameters
training:
  # Training epochs
  epochs: 5
  # (Optional, default=16) Depends on CPU-GPU available memory.
  batch_size: 16
  # (Optional, default=0.2) Train and validation split.
  validation_split: null
  # Models loss function
  loss:
    function: CategoricalCrossentropy
    parameters: null
  # (Optional, default=Adam(lr=1e-4)) Training optimizer
  optimizer:
    name: adam
    learning_rate: 1e-5
    parameters: null
  # Training callbacks
  callbacks:
    # Connect to weight and bias *Requiere plugins and env variable.* 
    enable_wandb: true
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
      log_predictions_plot: true
      log_predictions: true
      log_uncertainty_plot: true
      log_uncertainty: true
      log_confusion_matrix_plot: true
    # Save checkpoints: saved at experiments_folder/experiment/checkpoints
    checkpoints:
      # saved weights format
      format: "weights.{epoch:03d}-{val_loss:.4f}.h5"
      save_best_only: true
      save_weights_only: true
      monitor: val_loss
# Evaluation metrics
evaluation:
  metrics: [accuracy]
# Predict 
predict:
  save_aggregation: false
  save_predictions: true
  save_uncertainty: false
# Plugin setups
plugins:
  wandb:
    project: Her2BDL
    # Use environment variable name (recommended) or API KEY.
    #apikey: /data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl/.wandb_secret
    apikey: /home/asuka/projects/her2bdl/.wandb_secret
```


