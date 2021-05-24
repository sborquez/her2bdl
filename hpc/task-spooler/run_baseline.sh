#!/bin/bash
# ----------------Modules-----------------------------
project=$HER2BDL_HOME
experiments_folder=$HER2BDL_EXPERIMENTS
env="her2bdl"
# ----------------Comands--------------------------
jobs=10
experiments=(
    "$experiments_folder/config/baseline/hdxconv_c876_d02.yaml"
    "$experiments_folder/config/baseline/hdxconv_c876_d05.yaml"
    "$experiments_folder/config/baseline/hdxconv_c876_d08.yaml"
)
echo "Enqueue baseline tasks"
for i in $(seq 3 $jobs); do 
    for experiment in "${experiments[@]}"; do
        echo "Experment config file: $experiment"
        tsp sh -c "cd '$project/scripts';python train_model.py -c $experiment --quiet --dryrun --job $i";
        #tsp sh -c "cd '$project/scripts';python train_model.py -c $experiment --quiet --job $i";
    done
done

