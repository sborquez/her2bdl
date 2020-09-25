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

### This repository structure

#### train/

Collections of scripts for setup your data, config preprocessing pipes, and train/evaluate your models.

In these folder you can find also:
* `experiments/`: Folder with experiments configurations and results. More details and name conventions in `train/experiments/README.md`.
* `debug.py`: Utils functions for check GPU support, profiling keras/tensorflow models and plots models.
* `hpc/`: Script and instructions for run experiments in HPC environments.

More details for `train/` folder in  `train/README.md`.

#### notebooks/

Contains notebook version of the scripts in `\train`. More details in `notebooks/README.md`.

Handy for `Google Colab`.

#### test/

Collections of user defined unit tests with `nose`.

Here a tutorial for [nose](https://pythontesting.net/framework/nose/nose-introduction/) oython testing.

More details in `tests/README.md`.

#### deploy/

Different options for consume a trained model.

More details in `deploy/README.md`.


### Environments


Check anaconda documentation [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

#### Create environment

* `environment.yml`: for GPU support
* `environment_cpu.yml`: for CPU only support

```
conda env create -f environment.yml --name her2bdl
```


#### Load environment

```
conda activate her2bdl
```



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

