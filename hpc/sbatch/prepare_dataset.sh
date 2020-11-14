#!/bin/bash
#SBATCH -p <partition name>
#SBATCH -J her2bdl_dataset
#SBATCH --mail-user=sebastian.borquez.g@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o output_dataset_%j.log
#SBATCH -e error_dataset_%j.log

# ----------------Modules-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Variables--------------------------
project=~~/her2bdl
# ----------------Comands--------------------------

source activate "$project/.envs"
echo "Running prepare_dataset.sh"
echo ""

cd "$project/train"

# Setup datasets splits or any setup necesary
echo "Prepare Dataset Train"
python prepare_dataset.py -o "$project/dataset/training" -i <path to raw train data>  --split 0.1

# Test
echo "Preprocessing Test"
python prepare_dataset.py -o "$project/dataset/training" -i <path to test data> -s 0
