#!/bin/bash
# ----------------Variables--------------------------
project=/home/asuka/projects/her2bdl
env="her2bdl"
# ----------------Comands--------------------------

conda activate "$env"
echo "Running test_gpu.sh"
echo ""

cd "$project/train"
python debug.py --gpu