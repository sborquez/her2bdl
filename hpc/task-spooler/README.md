# Setup Instructions for running with Task-Spooler task queue manager


## Creating conda enviroment

First install Anaconda and task-spooler:

```
sudo apt install task-spooler
```

Create new enviroment and install dependencies.

```bash
conda env create -f <path to enviroment.yml>
```

## Running Scripts

Send jobs to queue with

```
tsp sh -C "cd /her2dbl/train; python train.py <args>"
```

*Note*: Send a jobs with an enviroment activated!


List jobs

```
tsp -l
```