#!/bin/bash
# ----------------Variables--------------------------
project=/data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl
jobs=1
overwrite_seed=false
experiments=(
    # HER2 Baseline
    ## Aleatoric
    #"$project/train/experiments/config/aleatoric/hdxconv_c876.yaml"
    #"$project/train/experiments/config/aleatoric/hedconv_c876.yaml"
    #"$project/train/experiments/config/aleatoric/rgbconv_c876.yaml"

    # Binary Classification
    ## Aleatoric
    "$project/train/experiments/config/aleatoric/efficientnet_b0_binary_classifier.yaml"
    #"$project/train/experiments/config/aleatoric/efficientnet_b1_binary_classifier.yaml"
    #"$project/train/experiments/config/aleatoric/efficientnet_b2_binary_classifier.yaml"
)

# ----------------Comands--------------------------
cd "$project/hpc/sbatch"
echo "Enqueue multiples train_model.sh jobs:"
for i in $(seq 1 $jobs); do 
    for experiment in "${experiments[@]}"; do
        echo "Experment config file: $experiment - Job $i enqueued"
        if $overwrite_seed; then
            tmp=${experiment#*d0}
            jobname="h2d${tmp:0:1}j$i"
            sbatch -J $jobname --export=job=$i,experiment=$experiment,seed=$((999 + $RANDOM % 10000)) train_model_aleatoric.sh
        else
            sbatch --export=job=$i,experiment=$experiment train_model_aleatoric.sh
        fi
    done
done
