#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J her2bdl_debug
#SBATCH --mail-user=sebastian.borquez.g@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o /user/s/sborquez/logs/her2bdl_debug_%j_output.log
#SBATCH -e /user/s/sborquez/logs/her2bdl_debug_%j_error.log 
#SBATCH --gres=gpu:1

# ----------------Moduls-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Variables--------------------------
project=/data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl
env="$project/.env"
# ----------------Comands--------------------------

source activate "$env"
echo "Running test_gpu.sh"
echo ""

cd "$project/train"
python debug.py --gpu