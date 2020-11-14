# Her2BDL

Bayesian Deep Learning for HER2 tissue image classification.
Her2BDL [Weights and Bias page](https://wandb.ai/sborquez/her2bdl)

## Features

- [x] Anaconda Envirment.
- [x] AI project structure.
- [x] Tensorflow, Keras and Scikit-learn.
- [x] HPC jobs manager.
- [x] Nose unit tests.
- [x] Weight and Bias easy setup.
- [ ] Easy experiment reproducibility.
- [ ] Notebooks.
- [ ] Singularity container.
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

###### Openslide will be installed by conda using the conda-forge channel.

##### Load environment

```
conda activate her2bdl
```

#### 1. Prepare Dataset

#### 2. Train Model with `Weight & Biases`

#### 3. Consume Models

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


### Bibliography

- GAL, Yarin. Uncertainty in deep learning. University of Cambridge, 2016, vol. 1, no 3.[http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)

### Dataset

- HER2 challenge contest: [https://onlinelibrary.wiley.com/doi/10.1111/his.13333](https://onlinelibrary.wiley.com/doi/10.1111/his.13333)

### Web

- YastAI Template: [https://github.com/yast-ia/YastAI](https://github.com/yast-ia/YastAI)
- WSL Preprocessing: [https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/](https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/)
- Weight and Bias: [https://www.wandb.com/](https://www.wandb.com/)
- Tensorflow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Tensorflow Probability: [https://www.tensorflow.org/probability](https://www.tensorflow.org/probability)
- Tensorflow Datasets:[https://www.tensorflow.org/datasets](https://www.tensorflow.org/datasets?hl=es-419)

