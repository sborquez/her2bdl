#!/bin/bash
#SBATCH -p <partition name>
#SBATCH -J yastai_train
#SBATCH --mail-user=yastaidata@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o output_yastai_train_%j.log
#SBATCH -e error_yastai_train_%j.log
#SBATCH --gres=gpu:2
#SBATCH --time=14-0

# ----------------Modules-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Variables--------------------------
project=~/YastAI
# ----------------Comands--------------------------

source activate "$project/.envs"
echo "Running train_model.sh"
echo ""
 
cd "$project/train"

## Parameters
experiment_results_folder="$project/train/experiments/runs"
experiment_config_file="$project/train/experiments/config/simple.json"

#TODO: Support cmd parameters
python train_model.py --experiment=$experiment_config_file --results_folder=$experiment_results_folder --quiet --multigpu
