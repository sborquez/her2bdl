# Setup Instructions for running in HPC with Slurm Workload Manager
## Deprecation Note

This maybe doesn't work. Required update the scripts by adding environment
variable support.

## Creating conda enviroment

First load Anaconda:

```
source /opt/software/anaconda3/2019.03/setup.sh
```

or if the HPC uses `Lmod` modules system:

```
ml  CUDA/10.1.105 HDF5/1.10.4   Miniconda3/4.5.12   cuDNN/7.4.2.24  
```

Create new enviroment in user level directory and install dependencies.

```bash
TODO
conda create -f <path to enviroment.yml> --prefix ./.env
```

~~conda create --prefix ./.env python=3.8 && conda activate "~/YastAI/.env"~~

## Running Scripts

Use this template to generate your scripts. If you want to reuse the existing files. Check the parameters and names of your projec and HPC parameters also.

Add this generated script at the begining of the sh file.

```bash
#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH -p gpuk <you can use gpum too, but it`s slower>
#SBATCH -J <job name here>
#SBATCH --mail-user=<your email here>
#SBATCH --mail-type=ALL
#SBATCH -o <your output log name>_%j.log
#SBATCH -e <your error log name>_%j.log
#SBATCH --gres=gpu:1

# ----------------Modules-----------------------------
cd $SLURM_SUBMIT_DIR

source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Variables--------------------------
project=~/YastAI
# ----------------Comands--------------------------
source activate "$project/.envs"
echo "Running <python script>.sh"
echo ""

cd "$project/train"
python <python script> <parameters>
```

Send jobs to queue with

```
sbatch <job>.sh
```

*Note*: never send a job with an enviroment activated!


List jobs

```
squeue
```

List nodes

```
sinfo
```




