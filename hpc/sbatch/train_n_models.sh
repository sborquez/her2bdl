#!/bin/bash
# ----------------Variables--------------------------
project=/data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl
jobs=3
experiments=(
    "$project/train/experiments/config/binary_classification/efficientnet_b0_binary_classifier.yaml"
    "$project/train/experiments/config/binary_classification/efficientnet_b1_binary_classifier.yaml"
    "$project/train/experiments/config/binary_classification/efficientnet_b2_binary_classifier.yaml"
    "$project/train/experiments/config/binary_classification/efficientnet_b3_binary_classifier.yaml"
    "$project/train/experiments/config/binary_classification/efficientnet_b4_binary_classifier.yaml"
)
# ----------------Comands--------------------------
cd "$project/hpc/sbatch"
echo "Enqueue multiples train_model.sh jobs:"
for experiment in "${experiments[@]}"; do
    echo "Experment config file: $experiment"
    for i in $(seq 1 $jobs); do 
        sbatch --export=job=$i,experiment=$experiment train_model.sh
        echo "- Job $i enqueued"
    done
done