#!/bin/bash
# ----------------Modules-----------------------------
project=$HER2BDL_HOME
experiments=$HER2BDL_EXPERIMENTS
env="her2bdl"
# ----------------Comands--------------------------
jobs=3
experiments=(
    "$experiments/config/aleatoric/hdxconv_c876.yaml"
    "$experiments/config/aleatoric/hedconv_c876.yaml"
    "$experiments/config/aleatoric/rgbconv_c876.yaml"
)
echo "Enqueue baseline tasks"
for i in $(seq 1 $jobs); do 
    for experiment in "${experiments[@]}"; do
        echo "Experment config file: $experiment"
        tsp sh -c "cd '$project/scripts';python train_aleatoric.py -c $experiment --quiet --dryrun --job $i";
        #tsp sh -c "cd '$project/scripts';python train_aleatoric.py -c $experiment --quiet --job $i";
    done
done

