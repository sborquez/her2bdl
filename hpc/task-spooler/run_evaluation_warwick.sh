#!/bin/bash
# ----------------Modules-----------------------------
project=$HER2BDL_HOME
experiments_folder=$$HER2BDL_HOME/scripts/config
env="her2bdl"
# ----------------Comands--------------------------
experiments=(
    # Classification
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d02_base.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d05_base.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d08_base.yaml"
    # Mixture
    ## Predictive Entropy
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d02_mixture_H.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d05_mixture_H.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d08_mixture_H.yaml"
    ## Mutual Information
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d02_mixture_I.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d05_mixture_I.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d08_mixture_I.yaml"
    # Threshold
    ## Predictive Entropy
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d02_threshold_Hh.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d02_threshold_Hl.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d05_threshold_Hh.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d05_threshold_Hl.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d08_threshold_Hh.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d08_threshold_Hl.yaml"
    ## Mutual Information
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d02_threshold_Ih.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d02_threshold_Il.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d05_threshold_Ih.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d05_threshold_Il.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d08_threshold_Ih.yaml"
    "$experiments_folder/baseline_evaluation_warwick/hdxconv_c876_d08_threshold_Il.yaml"

)
echo "Enqueue evaluation_warwick tasks"
for experiment in "${experiments[@]}"; do
    echo "Experment config file: $experiment"
    #tsp sh -c "conda activate '$env';cd '$project/scripts';python train_model.py -c $experiment --quiet --dryrun";
    tsp sh -c "conda activate '$env';cd '$project/scripts';python evaluate.py -c $experiment --quiet";
done

