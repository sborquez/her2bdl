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
experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b0_binary_classifier.yaml"
#experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b1_binary_classifier.yaml"
#experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b2_binary_classifier.yaml"
#experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b3_binary_classifier.yaml"
#experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b4_binary_classifier.yaml"
# ----------------Comands--------------------------
source activate "$project/.env"
echo "Running train_model.sh"
echo ""

if [ -z "$experiment" ];
then
    experiment=$experiment_config_file;
fi
cd "$project/train"
echo "Running $experiment"

if [ -z "$job" ];
then
    python train_model.py -c $experiment --quiet;
else
    python train_model.py -c $experiment --quiet --job $job;
fi