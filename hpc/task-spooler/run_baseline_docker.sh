#!/bin/bash
# ----------------Modules-----------------------------
#export HER2BDL_HOME=/mnt/storage-lite/archived_projects/her2bdl
project=$HER2BDL_HOME
experiments_folder=$HER2BDL_HOME/scripts/config
env="her2bdl"
# ----------------Comands--------------------------
jobs=1
experiments=(
    "$experiments_folder/baseline/hdxconv_c876_d02.yaml"
    "$experiments_folder/baseline/hdxconv_c876_d05.yaml"
    "$experiments_folder/baseline/hdxconv_c876_d08.yaml"
)
echo "Enqueue baseline tasks"
for i in $(seq 1 $jobs); do 
    for experiment in "${experiments[@]}"; do
        echo "Experment config file: $experiment"
        #tsp sh -c "cd '$project/scripts';python train_model.py -c $experiment --quiet --dryrun --job $i";
        #tsp sh -c "cd '$project/scripts';PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HER2BDL_HOME=/mnt/storage-lite/archived_projects/her2bdl HER2BDL_DATASETS=/mnt/storage-lite/archived_projects/her2bdl/scripts/datasets HER2BDL_EXPERIMENTS=/mnt/storage-lite/archived_projects/her2bdl/scripts/experiments HER2BDL_EXTRAS=/mnt/storage-lite/projects/her2bdl-files python train_model.py -c $experiment --quiet --job $i";
        tsp sh -c "cd '$project/scripts';python train_model.py -c $experiment --quiet --job $i";
    done
done

