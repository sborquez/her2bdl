# YastAI

Yet Another Session for Training Artificial Intelligence

This is a framework for easy development of new AI Research Projects.

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

There are 5 folder for develop  you new project:

#### AI/

`AI` folder is your AI project package,  for more details check `AI/README.md`

If you want to rename the `IA` folder with your project name, you must also edit theses files:

* `setup.py`: replace `ia_name` variable in __line 7__.
* `__version__.py` inside `AI` folder.
* Maybe you want to edit the conda envirment file `yastai.yml`.
* Replace `from AI import *` in every script in `train/`.
* Replace `from AI import *` in every nootebook in `notebooks/`.
* Replace `from AI import *` in every nootebook in `tests/`.
* Replace AI from files in `docs/`.
* ~~HPC train scripts maybe~~
* ~~conda environment maybe~~

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

