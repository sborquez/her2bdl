#!/bin/bash
# ----------------Modules-----------------------------
project=/home/asuka/projects/her2bdl
env="her2bdl"
# ----------------Comands--------------------------
jobs=10
experiments=(
    "$project/train/experiments/config/baseline/hdxconv_c876_d02.yaml"
    "$project/train/experiments/config/baseline/hdxconv_c876_d05.yaml"
    "$project/train/experiments/config/baseline/hdxconv_c876_d08.yaml"
)
echo "Enqueue baseline tasks"
for i in $(seq 3 $jobs); do 
    for experiment in "${experiments[@]}"; do
        echo "Experment config file: $experiment"
        tsp sh -c "cd '$project/train';python train_model.py -c $experiment --quiet --dryrun --job $i";
        #tsp sh -c "cd '$project/train';python train_model.py -c $experiment --quiet --job $i";
    done
done

