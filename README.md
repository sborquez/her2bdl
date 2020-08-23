# her2bdl

Bayesian Deep Learning for HER2 tissue image classification.

## Features

- [x] Anaconda Envirment.
- [x] AI project structure.
- [x] Tensorflow, Keras and Scikit-learn.
- [x] HPC jobs manager.
- [ ] Pip Package.
- [ ] Weight and Bias easy setup.
- [ ] Notebooks.
- [ ] Nose unit tests.
- [ ] Singularity container.
- [ ] Easy experiment reproducibility.

## How to start a new project/research

 This is a guide to understand the zen of this repository. 

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

#### api/

Different options for consume a trained model.

More details in `api/README.md`.


### environments

TODO


#### Load environment

Check anaconda documentation [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

* `environment.yml`
```
conda env create -f environment.yml --prefix ./.env
```

#### Crete/Update environments

TODO
Manually add packages to `enviroment.yml` and `Pipfile`.

### Documentation

TODO

### Publish in PyPi

TODO

Requiere PyPi and TestPyPi accounts

[doc](https://packaging.python.org/tutorials/packaging-projects/)

```
conda activate ./.env
python setup.py sdist bdist_wheel
twine check ./*
```

#### Test PyPi


```
twine upload --repository testpypi dist/*
```


#### Upload in PyPi

```
twine upload dist/*
```
