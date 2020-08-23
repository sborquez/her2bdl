#!/bin/bash
#SBATCH -p <partition name>
#SBATCH -J yastai_debug
#SBATCH --mail-user=yastaidata@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o output_yastai_debug_%j.log
#SBATCH -e error_yastai_debug_%j.log 
#SBATCH --gres=gpu:1

# ----------------Moduls-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Variables--------------------------
project=~/YastAI
# ----------------Comands--------------------------

source activate "$project/.envs"
echo "Running test_gpu.sh"
echo ""

cd "$project/train"
python debug.py --gpu