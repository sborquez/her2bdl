#!/bin/bash
# ----------------Variables--------------------------
project=/data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl
jobs=10
overwrite_seed=false
experiments=(
    # HER2 Baseline
    ## Epistemic
    #"$project/train/experiments/config/baseline/hdxconv_c876_d02.yaml"
    #"$project/train/experiments/config/baseline/hdxconv_c876_d05.yaml"
    #"$project/train/experiments/config/baseline/hdxconv_c876_d08.yaml"

    # Binary Classification
    ## Epistemic
    "$project/train/experiments/config/binary_classification/efficientnet_b0_binary_classifier.yaml"
    #"$project/train/experiments/config/binary_classification/efficientnet_b1_binary_classifier.yaml"
    #"$project/train/experiments/config/binary_classification/efficientnet_b2_binary_classifier.yaml"
    #"$project/train/experiments/config/binary_classification/efficientnet_b3_binary_classifier.yaml"
    #"$project/train/experiments/config/binary_classification/efficientnet_b4_binary_classifier.yaml"
)

# ----------------Comands--------------------------
cd "$project/hpc/sbatch"
echo "Enqueue multiples train_model.sh jobs:"
for i in $(seq 2 $jobs); do 
    for experiment in "${experiments[@]}"; do
        echo "Experment config file: $experiment - Job $i enqueued"
        if $overwrite_seed; then
            tmp=${experiment#*d0}
            jobname="h2d${tmp:0:1}j$i"
            sbatch -J $jobname --export=job=$i,experiment=$experiment,seed=$((999 + $RANDOM % 10000)) train_model.sh
        else
            sbatch --export=job=$i,experiment=$experiment train_model.sh
        fi
    done
done
