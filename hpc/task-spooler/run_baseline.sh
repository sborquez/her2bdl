#!/bin/bash
# ----------------Modules-----------------------------
project=$HER2BDL_HOME
experiments_folder=$HER2BDL_HOME/scripts/config
env="her2bdl"
source ./nvidia.sh
# ----------------Comands--------------------------
jobs=10
experiments=(
    "$experiments_folder/baseline/hdxconv_c876_d02.yaml"
    "$experiments_folder/baseline/hdxconv_c876_d05.yaml"
    "$experiments_folder/baseline/hdxconv_c876_d08.yaml"
)
echo "Enqueue baseline tasks"
for i in $(seq 6 $jobs); do 
    for experiment in "${experiments[@]}"; do
        echo "Experment config file: $experiment"
        tsp sh -c "cd '$project/scripts';python train_model.py -c $experiment --quiet --dryrun --job $i";
        #tsp sh -c "cd '$project/scripts';python train_model.py -c $experiment --quiet --job $i";
    done
done

