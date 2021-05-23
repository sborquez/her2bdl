#!/bin/bash
# ----------------Modules-----------------------------
project=/home/asuka/projects/her2bdl
experiment_folder="$project/train/experiments/config/baseline_evaluation_warwick"
env="her2bdl"
# ----------------Comands--------------------------
experiments=(
    # Classification
    "$experiment_folder/hdxconv_c876_d02_base.yaml"
    "$experiment_folder/hdxconv_c876_d05_base.yaml"
    "$experiment_folder/hdxconv_c876_d08_base.yaml"
    # Mixture
    ## Predictive Entropy
    "$experiment_folder/hdxconv_c876_d02_mixture_H.yaml"
    "$experiment_folder/hdxconv_c876_d05_mixture_H.yaml"
    "$experiment_folder/hdxconv_c876_d08_mixture_H.yaml"
    ## Mutual Information
    "$experiment_folder/hdxconv_c876_d02_mixture_I.yaml"
    "$experiment_folder/hdxconv_c876_d05_mixture_I.yaml"
    "$experiment_folder/hdxconv_c876_d08_mixture_I.yaml"
    # Threshold
    ## Predictive Entropy
    "$experiment_folder/hdxconv_c876_d02_threshold_Hh.yaml"
    "$experiment_folder/hdxconv_c876_d02_threshold_Hl.yaml"
    "$experiment_folder/hdxconv_c876_d05_threshold_Hh.yaml"
    "$experiment_folder/hdxconv_c876_d05_threshold_Hl.yaml"
    "$experiment_folder/hdxconv_c876_d08_threshold_Hh.yaml"
    "$experiment_folder/hdxconv_c876_d08_threshold_Hl.yaml"
    ## Mutual Information
    "$experiment_folder/hdxconv_c876_d02_threshold_Ih.yaml"
    "$experiment_folder/hdxconv_c876_d02_threshold_Il.yaml"
    "$experiment_folder/hdxconv_c876_d05_threshold_Ih.yaml"
    "$experiment_folder/hdxconv_c876_d05_threshold_Il.yaml"
    "$experiment_folder/hdxconv_c876_d08_threshold_Ih.yaml"
    "$experiment_folder/hdxconv_c876_d08_threshold_Il.yaml"

)
echo "Enqueue evaluation_warwick tasks"
for experiment in "${experiments[@]}"; do
    echo "Experment config file: $experiment"
    #tsp sh -c "conda activate '$env';cd '$project/train';python train_model.py -c $experiment --quiet --dryrun";
    tsp sh -c "conda activate '$env';cd '$project/train';python evaluate.py -c $experiment --quiet";
done

