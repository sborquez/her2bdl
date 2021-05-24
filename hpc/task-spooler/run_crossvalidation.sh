#!/bin/bash
# ----------------Modules-----------------------------
project=$HER2BDL_HOME
experiments_folder=$HER2BDL_EXPERIMENTS
env="her2bdl"
# ----------------Comands--------------------------
experiments=(
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c777_d02_kfold_1_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c777_d02_kfold_2_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c777_d02_kfold_3_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c777_d02_kfold_4_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c777_d02_kfold_5_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c777_d02_kfold_1_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c777_d02_kfold_2_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c777_d02_kfold_3_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c777_d02_kfold_4_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c777_d02_kfold_5_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c777_d02_kfold_1_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c777_d02_kfold_2_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c777_d02_kfold_3_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c777_d02_kfold_4_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c777_d02_kfold_5_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c876_d02_kfold_1_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c876_d02_kfold_2_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c876_d02_kfold_3_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c876_d02_kfold_4_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hdxconv_c876_d02_kfold_5_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c876_d02_kfold_1_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c876_d02_kfold_2_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c876_d02_kfold_3_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c876_d02_kfold_4_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/hedconv_c876_d02_kfold_5_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c876_d02_kfold_1_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c876_d02_kfold_2_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c876_d02_kfold_3_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c876_d02_kfold_4_5.yaml"
    "$experiments_folder/config/hyperparameters/crossvalidation_v2/rgbconv_c876_d02_kfold_5_5.yaml"
)
echo "Enqueue crossvalidation_v2 tasks"
for experiment in "${experiments[@]}"; do
    echo "Experment config file: $experiment"
    #tsp sh -c "conda activate '$env';cd '$project/scripts';python train_model.py -c $experiment --quiet --dryrun";
    tsp sh -c "conda activate '$env';cd '$project/scripts';python train_model.py -c $experiment --quiet";
done

