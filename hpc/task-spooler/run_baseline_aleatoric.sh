#!/bin/bash
# ----------------Modules-----------------------------
project=/home/asuka/projects/her2bdl
env="her2bdl"
# ----------------Comands--------------------------
jobs=3
experiments=(
    "$project/train/experiments/config/aleatoric/hdxconv_c876.yaml"
    "$project/train/experiments/config/aleatoric/hedconv_c876.yaml"
    "$project/train/experiments/config/aleatoric/rgbconv_c876.yaml"
)
echo "Enqueue baseline tasks"
for i in $(seq 1 $jobs); do 
    for experiment in "${experiments[@]}"; do
        echo "Experment config file: $experiment"
        tsp sh -c "cd '$project/train';python train_aleatoric.py -c $experiment --quiet --dryrun --job $i";
        #tsp sh -c "cd '$project/train';python train_aleatoric.py -c $experiment --quiet --job $i";
    done
done

