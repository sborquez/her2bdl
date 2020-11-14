#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J her2bdl_train
#SBATCH --mail-user=sebastian.borquez.g@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o /user/s/sborquez/logs/her2bdl_train_%j_output.log
#SBATCH -e /user/s/sborquez/logs/her2bdl_train_%j_error.log
#SBATCH --gres=gpu:1
#SBATCH --time=14-0

# ----------------Modules-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Variables--------------------------
project=/data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl
env="$project/.env"
#experiment_config_file="$project/train/experiments/config/simple_binary_classifier.yaml"
experiment_config_file="$project/train/experiments/config/efficientnet_binary_classifier.yaml"
# ----------------Comands--------------------------
source activate "$project/.env"
echo "Running train_model.sh"
echo ""
 
cd "$project/train"

## Parameters
echo "Experiment: $experiment_config_file"

#TODO: Support cmd parameters
python train_model.py -c $experiment_config_file --quiet
