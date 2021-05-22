#!/bin/bash
# ----------------Modules-----------------------------
project=/home/asuka/projects/her2bdl
experiment_folder="$project/train/experiments/config/baseline_evaluation_warwick"
env="her2bdl"
# ----------------Comands--------------------------
experiments=(
    "$experiment_folder/hdxconv_c876_d05_base.yaml"
    "$experiment_folder/hdxconv_c876_d08_base.yaml"

    "$experiment_folder/hdxconv_c876_d02_mixture_H.yaml"
    "$experiment_folder/hdxconv_c876_d02_mixture_I.yaml"
    "$experiment_folder/hdxconv_c876_d05_mixture_H.yaml"
    "$experiment_folder/hdxconv_c876_d05_mixture_I.yaml"
    "$experiment_folder/hdxconv_c876_d08_mixture_H.yaml"
    "$experiment_folder/hdxconv_c876_d08_mixture_I.yaml"

    "$experiment_folder/hdxconv_c876_d02_base.yaml"

)
echo "Enqueue evaluation_warwick tasks"
for experiment in "${experiments[@]}"; do
    echo "Experment config file: $experiment"
    #tsp sh -c "conda activate '$env';cd '$project/train';python train_model.py -c $experiment --quiet --dryrun";
    tsp sh -c "conda activate '$env';cd '$project/train';python evaluate.py -c $experiment --quiet";
done

