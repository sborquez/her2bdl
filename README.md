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
- [x] Docker container.
- [x] Easy experiment reproducibility.
- [ ] Notebooks.
- [ ] Pip Package.

## Her2BDL Framework - Environment Setup
### A. With Dockers

Using dockers is the preferred way; it is the easiest way to develop and run HER2BDL. Dockers containers enable us to encapsulate the environment. For instructions to install `dockers` check their [website](https://docs.docker.com/get-docker/).

Build a new image from your local copy of this repository

```bash
docker build -q -f "Dockerfile" -t her2bdl:latest  
```

Once the build process is finished, run an development environment with:

```bash
docker run -it \
     -v <path/to/dataset_folder>:/datasets \
     -v <path/to/experiment_folder>:/experiments \
     -v "$(pwd)":/her2bdl_dev \
     [-e WANDB_APY_KEY=<wandb_secret>]\
     her2bdl:latest
```

Ignore the `-v .:/her2bdl_dev` option if you will only use the framework. The `-e WANDB_APY_KEY=<wandb_secret>` argument is optional, replace `<wandb_secret>` with you own api key from [Weight & Biases](https://docs.wandb.ai/library/public-api-guide#authentication).

#### GPU support with nvidia-dockers 

Nvidia allows us to use host's GPU inside dockers with their extension `nvidia-docker`.
For instructions to install `nvidia-docker` check their [website](https://developer.nvidia.com/blog/nvidia-docker-gpu-server-application-deployment-made-easy).

Then, simply replace `docker` with `nvidia-docker` and `"Dockerfile"` with `"Dockerfile.nvidia"`.

```bash
nvidia-docker build -q -f "Dockerfile.nvidia" -t her2bdl:latest  
nvidia-docker run -it \
     -v <path/to/dataset_folder>:/datasets \
     -v <path/to/experiment_folder>:/experiments \
     -v "$(pwd)":/her2bdl_dev \
     [-e WANDB_APY_KEY=<wandb_secret>]\
     her2bdl:latest
```

**note:** `nvidia-docker` doesn't work on Windows.

### B. Without Dockers

As an alternative, you can set up the environment on your own.

**Prerequirements**:

* Anaconda3/Miniconda3 - [more](https://www.anaconda.com/products/individual#Downloads)
* (Optional) Nvidia GPU Drivers and CUDA - [more](https://www.tensorflow.org/install/gpu#software_requirements)
#### Setup conda environment

Run the following commands:

```bash
conda env create -f <environment.yml>
conda activate her2bdl
```

Replace `<environment.yml>` with one of theses options:

* `environment_cpu.yml`: for CPU only support
* `environment_gpu.yml`: for GPU support

Check anaconda [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) 
for more details about conda environments.

**note**: `conda activate her2bdl` this is required for each new session.

## Her2BDL Framework - Quick Start

### 1. Prepare Dataset

### 2. Train Model with `Weight & Biases`

### 3. Consume Models

## Repository structure

### train/

Collections of scripts for setup your data, config preprocessing pipes, and train/evaluate your models.

In these folder you can also find:
* `experiments/`: Folder with experiments configurations and results. More details and name conventions in `train/experiments/README.md`.
* `dataset/`: Default preprocessed dataset folder.
* `hpc/`: Script and instructions for run experiments in HPC environments.
* `debug.py`: Utils functions for check GPU support, profiling keras/tensorflow models and plots models.

More details for `train/` folder in  `train/README.md`.

### notebooks/

Contains notebook version of the scripts in `\train` handy for `Google Colab`. More details in `notebooks/README.md`.

### test/

Collections of user defined unit tests with `nose`. More details in `tests/README.md`.

### deploy/

Different options for consume a trained model. More details in `deploy/README.md`.

## Documentation

...


## References


### Bibliography

- GAL, Yarin. Uncertainty in deep learning. University of Cambridge, 2016, vol. 1, no 3.[http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)

### Dataset

- HER2 challenge contest: [https://onlinelibrary.wiley.com/doi/10.1111/his.13333](https://onlinelibrary.wiley.com/doi/10.1111/his.13333)

### Web

- YastAI Template: [https://github.com/yast-ia/YastAI](https://github.com/yast-ia/YastAI)
- WSL Preprocessing: [https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/](https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/)
- Weight and Bias: [https://www.wandb.com/](https://www.wandb.com/)
- Tensorflow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Tensorflow Datasets:[https://www.tensorflow.org/datasets](https://www.tensorflow.org/datasets?hl=es-419)

