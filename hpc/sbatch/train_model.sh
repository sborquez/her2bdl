#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J her2bdl_baseline
#SBATCH --mail-user=sebastian.borquez.g@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o /user/s/sborquez/logs/her2bdl_train_%j_output.log
#SBATCH -e /user/s/sborquez/logs/her2bdl_train_%j_error.log
#SBATCH --gres=gpu:1
#SBATCH --time=14-0

# ----------------Modules-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Variables--------------------------
project=/data/atlas/dbetalhc/cta-test/gerumo/src/her2bdl
env="$project/.env"

## Binary Classification
#experiment_config_file="$project/train/experiments/config/wsi_generators/efficientnet_b0_her2_grid_flip.yaml"
#experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b0_binary_classifier.yaml"
#experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b1_binary_classifier.yaml"
#experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b2_binary_classifier.yaml"
#experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b3_binary_classifier.yaml"
#experiment_config_file="$project/train/experiments/config/binary_classification/efficientnet_b4_binary_classifier.yaml"

## Cross Validation
### Test
#experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_test.yaml"
### hed
#### 02
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d02_kfold_1_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d02_kfold_2_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d02_kfold_3_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d02_kfold_4_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d02_kfold_5_5.yaml"
#### 05
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d05_kfold_1_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d05_kfold_2_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d05_kfold_3_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d05_kfold_4_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d05_kfold_5_5.yaml"
#### 08
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d08_kfold_1_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d08_kfold_2_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d08_kfold_3_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d08_kfold_4_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hedconv_c77_d08_kfold_5_5.yaml"

### hdx
#### 02
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d02_kfold_1_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d02_kfold_2_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d02_kfold_3_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d02_kfold_4_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d02_kfold_5_5.yaml"
#### 05
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d05_kfold_1_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d05_kfold_2_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d05_kfold_3_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d05_kfold_4_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d05_kfold_5_5.yaml"
#### 08
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d08_kfold_1_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d08_kfold_2_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d08_kfold_3_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d08_kfold_4_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/hdxconv_c77_d08_kfold_5_5.yaml"

### rgb
#### 02
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d02_kfold_1_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d02_kfold_2_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d02_kfold_3_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d02_kfold_4_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d02_kfold_5_5.yaml"
#### 05
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d05_kfold_1_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d05_kfold_2_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d05_kfold_3_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d05_kfold_4_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d05_kfold_5_5.yaml"
#### 08
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d08_kfold_1_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d08_kfold_2_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d08_kfold_3_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d08_kfold_4_5.yaml"
## experiment_config_file="$project/train/experiments/config/hyperparameters/crossvalidation/rgbconv_c77_d08_kfold_5_5.yaml"

## BASELINE
#experiment_config_file="$project/train/experiments/config/baseline/hedconv_c77_d02.yaml"
#experiment_config_file="$project/train/experiments/config/baseline/hedconv_c77_d05.yaml"
#experiment_config_file="$project/train/experiments/config/baseline/hedconv_c77_d08.yaml"

# ----------------Comands--------------------------
source activate "$project/.env"
echo "Running train_model.sh"
echo ""

# Overwrite experiment with bash variable
if [ -z "$experiment" ];
then
    experiment=$experiment_config_file;
fi
cd "$project/train"
echo "Running $experiment"

if [ -z "$job" ];
then
    python train_model.py -c $experiment --quiet;
else
    if [ -z "$seed" ];
    then
        python train_model.py -c $experiment --quiet --job $job;
    else
        python train_model.py -c $experiment --quiet --job $job --seed $seed;
    fi
fi