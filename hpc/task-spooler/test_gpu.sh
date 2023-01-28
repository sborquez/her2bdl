#!/bin/bash
# ----------------Variables--------------------------
project=$HER2BDL_HOME
env="her2bdl"
source ./nvidia.sh
# ----------------Comands--------------------------

conda activate "$env"
echo "Running test_gpu.sh"
echo ""

cd "$project/scripts"
python debug.py --gpu