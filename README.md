# Her2BDL

Bayesian Deep Learning for HER2 tissue image classification.

## Features

- [x] Anaconda Envirment.
- [x] AI project structure.
- [x] Tensorflow, Keras and Scikit-learn.
- [x] HPC jobs manager.
- [ ] Nose unit tests.
- [ ] Weight and Bias easy setup.
- [ ] Notebooks.
- [ ] Singularity container.
- [ ] Easy experiment reproducibility.
- [ ] Pip Package.

## Her2BDL Package

### Quick Start

#### 0. Setup environment

Check anaconda documentation [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

##### Install dependencies in a new environment

* `environment.yml`: for GPU support
* `environment_cpu.yml`: for CPU only support

```
conda env create -f environment.yml
```

##### Load environment

```
conda activate her2bdl
```

#### 1. Prepare Dataset

#### 2. Train Model with `Weight & Biases`

#### 3. ...

### Repository structure

#### train/

Collections of scripts for setup your data, config preprocessing pipes, and train/evaluate your models.

In these folder you can also find:
* `experiments/`: Folder with experiments configurations and results. More details and name conventions in `train/experiments/README.md`.
* `dataset/`: Default preprocessed dataset folder.
* `hpc/`: Script and instructions for run experiments in HPC environments.
* `debug.py`: Utils functions for check GPU support, profiling keras/tensorflow models and plots models.

More details for `train/` folder in  `train/README.md`.

#### notebooks/

Contains notebook version of the scripts in `\train` handy for `Google Colab`. More details in `notebooks/README.md`.



#### test/

Collections of user defined unit tests with `nose`. More details in `tests/README.md`.

#### deploy/

Different options for consume a trained model. More details in `deploy/README.md`.

## Documentation

...


## Referencias

### Dataset

- HER2 challenge contest: [https://onlinelibrary.wiley.com/doi/10.1111/his.13333](https://onlinelibrary.wiley.com/doi/10.1111/his.13333)

### Papers

- 

### Web

- YastAI Template: [https://github.com/yast-ia/YastAI](https://github.com/yast-ia/YastAI)
- WSL Preprocessing: [https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/](https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/)
- Weight and Bias: [https://www.wandb.com/](https://www.wandb.com/)
- Tensorflow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Tensorflow Probability: [https://www.tensorflow.org/probability](https://www.tensorflow.org/probability)
- Tensorflow Datasets:[https://www.tensorflow.org/datasets](https://www.tensorflow.org/datasets?hl=es-419)

